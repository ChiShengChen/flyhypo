# flyhypo architecture — grounding & verification

> Status: the literature anti-hallucination stack (steps 1–5 below) is **implemented** via
> `paper-evidence`, alongside a set of **connectivity-side guards** and an **evaluation harness**
> added since (both documented at the end of this file). The "Current (v0)" section is kept as the
> starting point this design replaced. This doc is both the design rationale and a record of the
> defects the harness surfaced and how they were fixed.

flyhypo's core promise is that **every claim traces to a specific paper or connectivity number,
and nothing is fabricated**. That promise is only as strong as the *verification* behind it. This
document upgrades flyhypo's verification from "ask the same model to check itself" to a layered,
mostly-deterministic anti-hallucination stack — reusing [`paper-evidence`](https://github.com/ChiShengChen/paper-evidence)
(a domain-neutral literature-evidence + claim-grounding library) as the verification layer.

## Current (v0)

```
cell type ─▶ connectome.py ─▶ StructuralFingerprint ─┐
            literature.py ─▶ [LiteratureHit]         ├─▶ synthesize.py ─▶ Hypothesis
              (PubMed abstracts, keyword queries)     │     Gemini generate,
                                                       │     then Gemini verify (self-check)
                                                       └─────┘
```

Weaknesses the last round of work exposed:

- **Self-grading.** The verify pass is Gemini checking Gemini — a shared blind spot, not an
  independent check.
- **No verbatim tie.** A claim cites a paper id but is not pinned to a *verbatim quote*; paraphrase
  drift and invented numbers survive.
- **Citations unverified.** "Strip ids not in the evidence" ≠ "the cited PMID/DOI is real and not
  retracted." An id the model invented but *echoed into the evidence bundle* can slip through.
- **Confidence is a model opinion.** Tiering is LLM judgment, not a rule; over-claim and
  abstract-only depth aren't systematically capped.
- **Coverage unknown.** Queries are built once from the fingerprint — no saturation, snowball, or
  recall measurement.

## Target (v1): the anti-hallucination stack

Each claim becomes a **premise** (what is attributed to the literature/connectome, carrying a
verbatim quote + the ids/numbers it rests on) plus a **prediction** (the novel, falsifiable leap).
**Only the premise is verified; the prediction is the hypothesis and rides along, labelled novel.**

```mermaid
flowchart TD
    F[StructuralFingerprint] --> Q[query plan]
    L[literature hits] --> Q
    Q -->|saturation + snowball| L
    L --> G["Gemini: generate roles<br/>premise (quote + ids/numbers) + prediction"]
    G --> V1["verbatim quote re-grep<br/>(quote_gate)"]
    V1 --> V2["numbers-in-context"]
    V2 --> V3["subject named in quote?<br/>(focus_named — anti mis-attribution)"]
    V3 --> V4["cross-family judge: does the quote<br/>support the premise? (DeepSeek ≠ Gemini)"]
    V4 --> C["citation real & not retracted?<br/>(citation.py: Crossref/OpenAlex/PubMed/arXiv)"]
    C --> S["evidence_state grade<br/>Supported/Tension/Conflict/Unknown + confidence"]
    S --> H[Hypothesis: graded, grounded roles + falsifiable predictions]
```

### The guards (mostly deterministic; from `paper-evidence`)

| Guard | What it kills | Module |
|---|---|---|
| **Verbatim quote re-grep** | paraphrase drift — a claim whose "quote" isn't actually in the abstract | `quote_gate.verify_quote` |
| **Numbers-in-context** | a connectivity/stat number that isn't next to the text it supposedly comes from | `quote_gate` (num_window) |
| **Mis-attribution guard** | a real quote about the *wrong* cell type (MBON01 vs MBON12, EPG vs PEG) — the focus must be **named in its own quote** | `quote_gate.focus_named` (card `subject` = the cell type + aliases) |
| **Cross-family faithfulness** | the premise overstates the quote — judged by a **different model family** (DeepSeek), so it isn't Gemini grading Gemini | `quote_gate.make_judge` |
| **Citation reality + retraction** | an invented or **retracted** PMID/DOI grounding a hypothesis | `citation.verify_citation` |
| **Evidence grading** | over-confidence — deterministic tiers, not a model's mood | `evidence_state.classify` |

### Evidence grading rules (deterministic — a strong fit for flyhypo)

`evidence_state` grades each premise `Supported / Tension / Conflict / Unknown` + a confidence tier,
with three rules that map cleanly onto flyhypo's reality:

- **overclaim = positive mis-attribution only.** A quote that says the wrong thing is an overclaim;
  a paper simply *not mentioning* the cell is **not**.
- **silence → Unknown, never Conflict.** An abstract that doesn't discuss a role is Unknown — flyhypo
  must not read absence as contradiction.
- **abstract-only caps confidence ≤ Med.** flyhypo is *entirely* abstract-level for literature, so its
  literature-grounded premises should cap at Med by construction (matches the existing single-neuron
  `low` cap, generalized). Full-text deep-read (below) is what lifts that ceiling.

## Coverage: know when you've searched enough

`literature.py` gains a recall layer (`paper-evidence.recall`): build query variants from the
fingerprint (cell type + aliases + neuropils + top partners), **stop only after *k* zero-yield
batches** (saturation), **snowball** citations from the best hits, and **audit recall** against a
review's reference list. Reworded queries are handled by **semantic anchoring** (`semantic.py`,
embedding similarity) so a paper that never uses the neuPrint ROI code still surfaces.

## Full text when it's open access

For the open-access subset of hits, `deepread.py` (PDF/URL/PMC → verbatim cards) replaces
abstract-only evidence for those papers — which, per the grading rule above, is what allows a
premise to exceed Med confidence.

## Migration path (incremental, each a small PR)

1. ✅ Add `paper-evidence` as a dependency; wrap each generated role as a card `{subject, claim, quote, numbers}`.
2. ✅ Replace the same-model verify pass with `quote_gate.build(..., judge=make_judge())` (verbatim +
   numbers-in-context + mis-attribution + cross-family judge). Drop failing cards, don't "downgrade".
3. ✅ Run `citation.verify_citation` on every cited PMID/DOI; drop unverified, flag `RETRACTED`.
4. ✅ Replace LLM confidence with `evidence_state.classify`.
5. ✅ `recall` layer — saturation (`recall.saturated`) + snowball (`literature.snowball`, self-contained
   Semantic Scholar) + audit (`literature.recall_audit`); `semantic` re-ranking via `FLYHYPO_SEMANTIC`.
6. ◻︎ `deepread` for OA hits — `literature.deep_read_hit()` exists as an opt-in hook, but is NOT wired
   into the default flow (it departs from the abstracts-only policy — a deliberate design decision).

Steps 1–5 are done (main + hierarchy paths). Step 6 is an opt-in hook pending a call on whether to
relax the abstracts-only policy for the open-access subset. None required changing the connectome layer.

## Why reuse `paper-evidence`

It is the domain-neutral extraction of exactly this anti-hallucination machinery (built and hardened
on real literature, including fruit-fly papers), with the verification core as **stdlib, zero-dep,
offline**. flyhypo stays the *application* (connectome + fly domain + UI); `paper-evidence` is the
*verification library*. See its README for the full API.

## Connectivity-side guards (the structural counterpart)

`paper-evidence` verifies *literature* claims. Connectivity claims — the tool's strong suit — get
their own deterministic guards in flyhypo, because a connectome number is only trustworthy if it is
actually in the fingerprint:

- **`numverify.py` — number guard.** Every synapse-scale number a role/hypothesis cites in
  `connectivity_basis` / `supporting_structure` must appear in the `StructuralFingerprint` (or the
  cross-dataset replication weights); a number that is nowhere in the evidence is flagged and the
  claim downgraded one tier. Conservative (ignores integers < 100 and citation years). This is
  flyhypo's own guard, distinct from `paper-evidence`'s literature "numbers-in-context" check.
- **Connectivity salvage.** A claim with its own connectivity grounding is never *dropped* for a
  literature-quote miss — its unverified literature support is stripped and it is kept on the
  fingerprint numbers at low confidence. A real structural claim shouldn't die because a paraphrase
  failed.
- **Evidence-type inference.** A hypothesis that cites no literature is a *connectivity* claim, not a
  literature claim to drop for lacking a quote.
- **Unstudied-type humility cap.** If *nothing* in the report could be grounded in literature, the
  type is effectively unstudied — connectivity shows wiring, not validated function — so all claims
  cap at `low`.
- **Single-neuron cap.** For one bodyId, function is inherited from the type; cap at `low`.

## Evaluation-driven development (what the harness caught)

`flyhypo-eval` — a curated gold set over 9 well-characterised types (5 systems) plus **negative
cases** (obscure `SMP029`, invalid `SA1`) — is not just a score; it is the mechanism that surfaced
most of the guards above. Every fix below was **measured first, then made**:

| The harness showed | Root cause | Fix |
|---|---|---|
| recall 0.58; hypotheses dropped | connectivity-only hypotheses treated as literature claims, dropped for lacking a quote | evidence-type inference (connectivity ≠ literature) |
| EPG recall bounced 0.67 ↔ 1.00 | a *generation-coverage* gap — the output/relay role wasn't reliably emitted (**not** a verify drop) | prompt now covers input → computation → output |
| KCg-m NT = dopamine (wrong) | `predictedNt` (raw per-synapse ML) mis-labels whole classes | prefer curated `consensusNt`; add an **NT check to the harness** so this error class is caught, not eyeballed |
| precision / partner-coverage < 1.0 | over-strict **gold** (redundant `Kenyon`, class-level `DAN`, wrong keys) | calibrate the gold, not the tool |
| SMP029 rated `medium` | connectivity-only functional guesses for an unstudied type weren't capped | unstudied-type humility cap |

The discipline the project follows: **measure before you fix**, and when a low score is the *gold's*
fault (a synonym miss, an over-strict partner key), fix the gold — never tune the tool for the score.
Determinism helps this loop: structured generation runs at `temperature=0`, and the deterministic
guards (numverify, citation hygiene, the caps) make re-runs comparable.
