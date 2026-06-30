"""Synthesis layer: turn (fingerprint + literature) into a grounded Hypothesis.

Backed by Google Gemini (google-genai). One call produces the tiered, falsifiable
hypotheses; a second lightweight call verifies each statement against the
evidence bundle and flags anything unsupported. The system prompt IS the
product — it enforces grounding, tiering, the not-supported-by-connectivity
section, and "never fabricate".
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .schema import (
    Hypothesis,
    HypothesisAnalysis,
    LiteratureHit,
    StructuralFingerprint,
    VerificationResult,
)

load_dotenv()

# Gemini model. Override with FLYHYPO_MODEL if you want flash / a newer version.
MODEL = os.environ.get("FLYHYPO_MODEL", "gemini-2.5-pro")

SYSTEM_PROMPT = """\
You are a Drosophila connectomics analyst that proposes grounded, falsifiable \
functional hypotheses about a single fly neuron cell type, for experimentalists \
to test. You are given two evidence sources:
  (1) STRUCTURE: a connectome-derived structural fingerprint (neuPrint) — \
resolved cells, predicted neurotransmitter, top input/output ROIs with synapse \
counts, and top up/down-stream partner types with synapse weights.
  (2) LITERATURE: abstracts/metadata of real papers retrieved for this type, \
its neuropils, and its partners.

CORE PRINCIPLE — a connectome tells you who connects to whom and roughly how \
strongly, but NOT synapse sign, effective/intrinsic strength, or \
neuromodulation. Predicted neurotransmitter is a prediction, not ground truth. \
Connection weights also vary across individual flies. Therefore your output is a \
HYPOTHESIS for experimentalists, never a stated fact.

RULES:
1. GROUND EVERY CLAIM. Each hypothesis statement must reference either a \
specific fingerprint field WITH its number (e.g. "receives 412 synapses from \
cholinergic ER4m cells in EB") via supporting_structure, OR a specific paper id \
via supporting_literature. No ungrounded claims.
2. TIER EXPLICITLY and set confidence accordingly:
   - high/medium: structurally supported (strong, specific connectivity numbers) \
and/or directly literature-supported.
   - low: supported only by partner/region literature, or weak/indirect structure.
   - speculative: a reasonable mechanistic guess beyond the evidence.
3. ALWAYS populate not_supported_by_connectivity. At minimum note that synapse \
sign, effective strength, and neuromodulation are unknown from connectivity, and \
that weights vary across individuals — plus anything else your hypotheses assume \
that the connectome cannot establish.
4. Propose >=1 concrete falsification experiment per NON-speculative hypothesis \
(optogenetic activation/silencing, calcium imaging, behavioural assay, \
electrophysiology, etc.), each with an expected_result. Use hypothesis_ref like \
"H1", "H2" matching the order of your hypotheses.
5. If literature for this exact type is sparse or absent: SAY SO in caveats, fall \
back to partner/region literature at LOWER confidence, and NEVER invent citations \
or PMIDs. Only cite ids that appear in the provided LITERATURE list.
6. If the structural fingerprint is empty (type not found), produce no \
structural claims; rely on whatever literature exists at low/speculative \
confidence and state the limitation prominently in caveats.

Be specific and quantitative. Prefer fewer, well-grounded hypotheses over many \
vague ones."""

VERIFY_SYSTEM = """\
You are a strict verifier. You are given an EVIDENCE bundle (structural \
fingerprint + literature ids/snippets) and a set of proposed HYPOTHESES. For \
each hypothesis statement, check whether its supporting_structure numbers \
actually appear in the fingerprint and whether every cited literature id \
actually appears in the evidence. Flag: (a) any claim with no valid grounding, \
(b) any cited id NOT present in the evidence (possible fabrication), (c) any \
statement phrased as established fact rather than hypothesis, (d) any missing \
not_supported_by_connectivity caveat. Be concise and concrete.

Also populate confidence_adjustments: for any hypothesis whose stated confidence \
is HIGHER than its evidence warrants (phrased as established fact, weaker/indirect \
support than the tier implies, or grounding that does not hold up), add an entry \
with the 1-based hypothesis_index, a LOWER recommended_confidence, and a one-line \
reason. Only recommend downgrades, never upgrades. Leave the list empty if every \
tier is justified."""


def _evidence_bundle(fp: StructuralFingerprint, lit: list[LiteratureHit]) -> str:
    return json.dumps(
        {
            "structure": fp.model_dump(by_alias=True),
            "literature": [h.model_dump() for h in lit],
        },
        indent=2,
        default=str,
    )


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get one at https://aistudio.google.com "
            "and put it in .env (see .env.example)."
        )
    return genai.Client(api_key=api_key)


CONFIDENCE_RANK = {"speculative": 0, "low": 1, "medium": 2, "high": 3}


def _generate(client: genai.Client, system: str, prompt: str, schema):
    """One structured-output Gemini call with explicit reasoning enabled.

    Returns (validated pydantic instance | None, reasoning_summary). Thinking is
    left on a dynamic budget so the model reasons proportionally to difficulty;
    its thought summary is captured for transparency.
    """
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_budget=-1  # -1 = dynamic
            ),
        ),
    )
    thoughts: list[str] = []
    try:
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    thoughts.append(part.text)
    except Exception:
        pass
    return resp.parsed, "\n".join(thoughts).strip()


def synthesize(fp: StructuralFingerprint, lit: list[LiteratureHit]) -> Hypothesis:
    client = _client()
    bundle = _evidence_bundle(fp, lit)

    # --- pass 1: generate the grounded analysis ------------------------- #
    analysis, reasoning = _generate(
        client,
        SYSTEM_PROMPT,
        (
            f"Cell type: {fp.cell_type_query}\nDataset: {fp.dataset}\n\n"
            f"EVIDENCE:\n{bundle}\n\n"
            "Produce the structured hypothesis analysis."
        ),
        HypothesisAnalysis,
    )
    if analysis is None:
        raise RuntimeError("Synthesis produced no parseable output.")

    # --- deterministic citation hygiene: drop any cited id not in the
    #     retrieved evidence, so a fabricated/hallucinated id can never survive. #
    valid_ids = {h.id for h in lit}
    stripped: set[str] = set()
    for hyp in analysis.hypotheses:
        kept = [i for i in hyp.supporting_literature if i in valid_ids]
        stripped.update(set(hyp.supporting_literature) - set(kept))
        hyp.supporting_literature = kept

    # --- pass 2: verify each statement against the evidence ------------- #
    verification, _ = _generate(
        client,
        VERIFY_SYSTEM,
        (
            f"EVIDENCE:\n{bundle}\n\n"
            f"Valid literature ids: {sorted(valid_ids)}\n\n"
            f"PROPOSED HYPOTHESES:\n{analysis.model_dump_json(indent=2)}\n\n"
            "Return your verification."
        ),
        VerificationResult,
    )
    verification = verification or VerificationResult(
        verification_notes="(verification pass returned no output)"
    )

    # --- reasoning-driven gating: apply verifier downgrades (never upgrade) - #
    downgrades: list[str] = []
    for adj in verification.confidence_adjustments:
        i = adj.hypothesis_index - 1
        if not (0 <= i < len(analysis.hypotheses)):
            continue
        hyp = analysis.hypotheses[i]
        cur, rec = hyp.confidence, adj.recommended_confidence
        if CONFIDENCE_RANK.get(rec, 99) < CONFIDENCE_RANK.get(cur, 0):
            downgrades.append(f"H{adj.hypothesis_index} {cur}→{rec} ({adj.reason})")
            hyp.confidence = rec

    notes = verification.verification_notes
    if stripped:
        notes += (f"\n\n[auto] Removed {len(stripped)} cited id(s) absent from the "
                  f"evidence (anti-fabrication): {', '.join(sorted(stripped))}.")
    if downgrades:
        notes += "\n\n[auto] Confidence downgraded by verification: " + "; ".join(downgrades) + "."

    return Hypothesis(
        cell_type=fp.cell_type_query,
        dataset=fp.dataset,
        fingerprint=fp,
        literature=lit,
        hypotheses=analysis.hypotheses,
        not_supported_by_connectivity=analysis.not_supported_by_connectivity,
        proposed_experiments=analysis.proposed_experiments,
        caveats=analysis.caveats,
        verification_notes=notes,
        reasoning_summary=reasoning[:2000],
    )
