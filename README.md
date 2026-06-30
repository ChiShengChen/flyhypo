# flyhypo

**A Drosophila neuron functional-hypothesis generator (proof of concept).**

Given a fly cell type (e.g. `EPG`, `MBON01`), `flyhypo` combines **structural
evidence** from a connectome (neuPrint) with **functional evidence** from the
literature (PubMed), then asks an LLM (Google Gemini) to synthesise **the
functional roles the neuron is involved in — each grounded in a paper id and/or a
specific connectivity number** — plus tiered, falsifiable hypotheses. Output as
structured JSON + a readable Markdown report.

> **Core principle.** A connectome tells you *who connects to whom and roughly
> how strongly*, but **not** synapse sign, effective/intrinsic strength, or
> neuromodulation — and connection weights vary across individuals. So every
> output is a **hypothesis for experimentalists, never a stated fact**. Each
> claim traces to a specific connectivity number or a specific paper. When
> evidence is thin, the tool says so and lowers confidence — it never fabricates.

![Connectivity graph for EPG](docs/graph.png)

*Connectivity graph (from the web UI) for `EPG` — the structural evidence the
hypotheses are grounded in: top-8 upstream partners → EPG → top-8 downstream
partners, edge thickness ∝ synapse weight, with synapse counts and cell counts `n`.*

---

## How it works

```
cell type ─▶ connectome.py ─▶ StructuralFingerprint ─┐
              (neuPrint)                              ├─▶ synthesize.py ─▶ Hypothesis
            literature.py ─▶ [LiteratureHit] ─────────┘   (Gemini ×2:           (JSON + .md)
              (PubMed)        (queries built from               generate, then
                               the fingerprint)                 verify)
```

| Module | Responsibility |
|---|---|
| `schema.py` | Pydantic data contracts (`StructuralFingerprint`, `LiteratureHit`, `Hypothesis`). |
| `connectome.py` | Pure, typed wrapper over `neuprint-python`. Resolves a type → fingerprint; fuzzy-suggests on miss. *(Designed to be liftable into a standalone MCP server.)* |
| `literature.py` | Builds queries from the fingerprint and retrieves abstracts/metadata via `paper-search-mcp`. |
| `synthesize.py` | Gemini (with explicit reasoning) generates tiered hypotheses; a verification pass then **downgrades** any over-stated confidence and the pipeline **strips any cited id not in the evidence**. The model's thought summary is surfaced as `reasoning_summary`. |
| `cli.py` | `flyhypo <cell_type> …` → writes `<cell_type>.json` + `.md`, prints a summary. |

---

## Prerequisites

- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)**.
- A **neuPrint token** — log in at <https://neuprint.janelia.org> (Google
  account), then *your name (top-right) → Account* and copy the auth token. The
  default dataset `hemibrain:v1.2.1` needs no special access.
- A **Google Gemini API key** — <https://aistudio.google.com> → *Get API key*.

PubMed needs no key.

## Setup

```bash
uv sync                 # install deps
cp .env.example .env    # then paste your two tokens into .env
```

## Run

```bash
# Full pipeline (needs both tokens):
uv run flyhypo EPG
uv run flyhypo MBON01 --dataset hemibrain:v1.2.1 --top-k 15 --out outputs/

# Just the structural fingerprint — only needs the neuPrint token:
uv run flyhypo EPG --fingerprint-only

# Unknown type → graceful fuzzy suggestions, no crash:
uv run flyhypo SA1 --fingerprint-only

# Single-neuron mode — structural fingerprint for one bodyId:
uv run flyhypo --neuron 387364605

# Multi-level analysis — region ▸ subregion ▸ umbrella ▸ cell type ▸ neuron:
uv run flyhypo EPG --hierarchy
uv run flyhypo --neuron 387364605 --hierarchy
```

Outputs land in `outputs/<cell_type>.json` and `outputs/<cell_type>.md`.

> **Tip — pass a cell *type*, not a brain region.** flyhypo resolves cell *types*.
> Neuropil / region names like `MB` (mushroom body), `EB`, `FB`, `PB` are **ROIs**,
> not types — so querying one resolves to no cells. The tool detects this, says so,
> and suggests representative cell types that arborize in that region (e.g. `MB` →
> Kenyon cells `KCg-m`, `KCab-c`, …; plus the 36 `MBON…` output neurons). Pass an
> actual type instead: `KCg-m`, `MBON01`, `EPG`. Unknown/typo'd types fall back to
> fuzzy name suggestions.

> **Tip — what the numbers mean.** ROI tables show **synaptic *site* counts**
> (pre/post sites summed over the cells); partner `w` / "synapse count" is the
> **pairwise synapse count** (synapses between neuron pairs, summed over the type).
> Both are structural proxies — not functional strength, sign, or reliability.

> **Single-neuron mode & its limits.** `--neuron <bodyId>` (or a numeric query in
> the web UI) fingerprints one individual neuron. The **structure** is genuinely
> single-cell — its own partners, ROIs, and **topographic position** (instance +
> sub-compartments, e.g. EB wedges / the PB glomerulus). But a single cell's
> **function is inherited from its type**; its only cell-specific signal is *where
> in the map* it sits. So hypotheses are **capped at `low` confidence** and the
> report states the hard gaps explicitly: there is **no literature for an
> individual neuron** (citations are type/region-level), connectivity is from
> **one fly (n=1)** so a cell's wiring can't be separated from individual /
> reconstruction idiosyncrasy, and synapse sign/strength/neuromodulation stay
> unknown. Closing those needs cross-individual data (FlyWire) + single-cell
> physiology — out of scope here.

> **Multi-level mode (`--hierarchy`).** Reports functional roles at **every level**
> of the hierarchy, each role grounded in paper id(s) and/or connectivity numbers:
> **region** (neuropil, e.g. `EB`) ▸ **subregion** (compartment, e.g. `EBr2r4`) ▸
> **umbrella** (the functional *system* / type-family — **named by the model**,
> e.g. "central-complex compass system (EPG/PEN/PEG)", grounded in refs + shared
> wiring) ▸ **cell type** (`EPG`) ▸ **neuron** (a bodyId, capped at `low`).
> Region/subregion come from neuPrint ROIs and the type's dominant compartments;
> coarser levels describe the region/system as a whole, not the single cell. Tip:
> run once to warm the cache — the first uncached run makes many live neuPrint
> calls and can be slow on a flaky connection.

### Web UI

A minimal local web UI (stdlib-only, same `.env` tokens) wraps the CLI:

```bash
uv run flyhypo-web            # → http://127.0.0.1:8000
```

![flyhypo web UI — full EPG report](docs/screenshot.png)

Type a cell type, pick **Full** or **Fingerprint only**, and the page renders:

- **functional roles** — the headline answer: the functions this neuron is
  implicated in, each tagged literature / connectivity / both, with the paper ids
  and connectivity numbers that ground it;
- the **structural fingerprint** (input/output ROI tables, up/down-stream partner tables);
- a **connectivity graph** (inline SVG) — the target in the centre, upstream partners
  on the left and downstream on the right, edge thickness ∝ synapse weight, with the
  synapse count and cell-count `n` labelled and full details on hover (top 8 each side);
- **tiered hypotheses** colour-coded by confidence, linked literature (PMID/DOI →
  pubmed/doi.org), proposed experiments, the not-supported section, and verification notes.

Other niceties:

- **⬇ JSON / ⬇ Markdown** download buttons on each result.
- A **Saved reports** strip (everything in `outputs/`) with relative timestamps; click to
  re-open a past report, or **✕** to delete it. Full runs auto-save like the CLI.
- Not-found types show **clickable fuzzy suggestions**.

Everything is stdlib-only and offline-friendly (no external JS/CSS); the graph is
hand-built SVG. Partners are aggregated **by type** (type + cell-count + total synapses),
matching the connectome layer — not individual `bodyId`s.

### Flags

| Flag | Meaning |
|---|---|
| `--dataset` | neuPrint dataset (default `hemibrain:v1.2.1`). |
| `--top-k` | Number of up/down-stream partner types to keep (default 15). |
| `--out` | Output directory (default `outputs/`). |
| `--neuron BODYID` | Single-neuron mode: fingerprint one bodyId instead of a cell type. |
| `--hierarchy` | Analyze every level (region ▸ subregion ▸ umbrella/system ▸ cell type ▸ neuron), each with functional roles + refs. |
| `--fingerprint-only` | Stop after step 1 (no Gemini key needed). |
| `--no-cache` | Bypass the on-disk query cache. |

Results from neuPrint and PubMed are cached under `.flyhypo_cache/` (wipe with
`rm -rf .flyhypo_cache`). Set `FLYHYPO_USE_BIORXIV=1` to also query bioRxiv (its
searcher is noisy for fly queries, so it's off by default).

---

## Scope

**In scope (built):** one target cell type per run; structure from neuPrint;
literature from PubMed via `paper-search-mcp`; synthesis via the Google Gemini API;
structured JSON + Markdown output; on-disk caching; graceful degradation on
unknown types.

**Out of scope (clearly-marked TODOs — *not* built):**

- [ ] Cross-dataset replication (e.g. confirm a motif in FlyWire as well as hemibrain).
- [ ] FlyWire / Codex local DB. *(The later path is FlyWire/Codex static CSV + CAVE — **we do not scrape Codex**; this PoC uses the neuPrint API only.)*
- [ ] Virtual Fly Brain integration.
- [x] Web UI — a minimal local one ships (`flyhypo-web`); a hosted/multi-user version is still out of scope.
- [ ] Batch / evaluation harness.

## Guardrails honoured

- Uses the **neuPrint API**, not Codex scraping.
- Literature: **abstracts/metadata only**, never full text.
- Nothing about a cell's biology is hardcoded — everything comes from live
  queries + retrieved literature. (The only static table is a generic
  neuropil-abbreviation glossary used to widen search recall — region metadata,
  not cell-specific knowledge.)
- Unknown cell type → fuzzy suggestions + low-confidence degraded output, no
  invented references.
