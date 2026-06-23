"""Pydantic data contracts for flyhypo.

These models are the typed boundary between the connectome layer, the
literature layer, and the LLM synthesis layer. Keeping them pure and typed is
what would let `connectome.py` later be lifted into a standalone MCP server.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["high", "medium", "low", "speculative"]


# --------------------------------------------------------------------------- #
# Structural evidence (from neuPrint)
# --------------------------------------------------------------------------- #
class ResolvedInstance(BaseModel):
    bodyId: int
    type: str | None = None
    instance: str | None = None


class RoiWeight(BaseModel):
    roi: str
    weight: int


class Partner(BaseModel):
    """An up- or down-stream partner *type* (aggregated over its cells)."""

    model_config = ConfigDict(populate_by_name=True)

    type: str | None = None
    n_cells: int
    total_weight: int
    predicted_nt: str | None = None
    # 'class' is a Python keyword; expose it under the contract name via alias.
    neuron_class: str | None = Field(default=None, alias="class")


class StructuralFingerprint(BaseModel):
    cell_type_query: str
    dataset: str
    resolved: list[ResolvedInstance] = Field(default_factory=list)
    predicted_nt: str | None = None
    input_rois: list[RoiWeight] = Field(default_factory=list)
    output_rois: list[RoiWeight] = Field(default_factory=list)
    upstream: list[Partner] = Field(default_factory=list)
    downstream: list[Partner] = Field(default_factory=list)
    # Populated only when the type was NOT found, so the caller can degrade
    # gracefully instead of crashing.
    suggestions: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def found(self) -> bool:
        return len(self.resolved) > 0


# --------------------------------------------------------------------------- #
# Functional evidence (from the literature)
# --------------------------------------------------------------------------- #
class LiteratureHit(BaseModel):
    title: str
    source: str  # 'pubmed' | 'biorxiv' | ...
    id: str  # PMID / DOI / arXiv id
    year: int | None = None
    snippet: str  # abstract snippet ONLY — never full text
    relevance: str  # why this paper was retrieved


# --------------------------------------------------------------------------- #
# Synthesis output (from the LLM) + final assembled report
# --------------------------------------------------------------------------- #
class HypothesisItem(BaseModel):
    statement: str
    rationale: str
    # Grounding: each entry cites a specific fingerprint field WITH its number,
    # e.g. "input_rois: EB=412 synapses" or "upstream: ER4m (n=18, w=903)".
    supporting_structure: list[str] = Field(default_factory=list)
    # Literature ids (PMID/DOI) that back this statement.
    supporting_literature: list[str] = Field(default_factory=list)
    confidence: Confidence


class ProposedExperiment(BaseModel):
    hypothesis_ref: str  # e.g. "H1" — references the statement it would test
    method: str  # optogenetics / calcium imaging / behavioural assay / ...
    expected_result: str


class HypothesisAnalysis(BaseModel):
    """The part the LLM produces. Fingerprint + literature are attached by us."""

    hypotheses: list[HypothesisItem] = Field(default_factory=list)
    not_supported_by_connectivity: list[str] = Field(default_factory=list)
    proposed_experiments: list[ProposedExperiment] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Lightweight second-pass check of each statement against the evidence."""

    unsupported_claims: list[str] = Field(default_factory=list)
    verification_notes: str


class Hypothesis(BaseModel):
    """The final, schema-valid object written to disk."""

    cell_type: str
    dataset: str
    fingerprint: StructuralFingerprint
    literature: list[LiteratureHit] = Field(default_factory=list)
    hypotheses: list[HypothesisItem] = Field(default_factory=list)
    not_supported_by_connectivity: list[str] = Field(default_factory=list)
    proposed_experiments: list[ProposedExperiment] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    verification_notes: str = ""
