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
not_supported_by_connectivity caveat. Be concise and concrete."""


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


def _generate(client: genai.Client, system: str, prompt: str, schema):
    """One structured-output Gemini call → validated pydantic instance (or None)."""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return resp.parsed


def synthesize(fp: StructuralFingerprint, lit: list[LiteratureHit]) -> Hypothesis:
    client = _client()
    bundle = _evidence_bundle(fp, lit)

    # --- pass 1: generate the grounded analysis ------------------------- #
    analysis = _generate(
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

    # --- pass 2: verify each statement against the evidence ------------- #
    valid_ids = sorted({h.id for h in lit})
    verification = _generate(
        client,
        VERIFY_SYSTEM,
        (
            f"EVIDENCE:\n{bundle}\n\n"
            f"Valid literature ids: {valid_ids}\n\n"
            f"PROPOSED HYPOTHESES:\n{analysis.model_dump_json(indent=2)}\n\n"
            "Return your verification."
        ),
        VerificationResult,
    ) or VerificationResult(verification_notes="(verification pass returned no output)")

    return Hypothesis(
        cell_type=fp.cell_type_query,
        dataset=fp.dataset,
        fingerprint=fp,
        literature=lit,
        hypotheses=analysis.hypotheses,
        not_supported_by_connectivity=analysis.not_supported_by_connectivity,
        proposed_experiments=analysis.proposed_experiments,
        caveats=analysis.caveats,
        verification_notes=verification.verification_notes,
    )
