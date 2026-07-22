"""flyhypo as an MCP server — expose the connectome hypothesis engine as tools
other agents (or Claude) can call.

    uv run flyhypo-mcp          # stdio server

The connectome layer was designed to be liftable into a standalone MCP server;
this is that. Structure tools need only a neuPrint token (NEUPRINT_APPLICATION_
CREDENTIALS); `hypothesize` also needs GEMINI_API_KEY. Every claim traces to a
connectivity number or a paper — output is a hypothesis for experimentalists,
not a stated fact.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import connectome, literature, replication, synthesize

DATASET = connectome.DEFAULT_DATASET

mcp = FastMCP(
    "flyhypo",
    instructions=(
        "Drosophila neuron functional-hypothesis engine over the neuPrint connectome. "
        "A connectome gives connectivity — not synapse sign, effective strength, or "
        "neuromodulation — so everything is a hypothesis for experimentalists, never a "
        "stated fact. Structure tools (fingerprint, neuron_fingerprint, replicate) need "
        "only a neuPrint token; hypothesize also needs GEMINI_API_KEY."
    ),
)


@mcp.tool()
def fingerprint(cell_type: str, dataset: str = DATASET, top_k: int = 15) -> dict:
    """Structural fingerprint of a fly cell TYPE from the connectome (neuPrint, no LLM).

    Returns resolved cells, predicted neurotransmitter (borrowed from a sibling
    connectome if the dataset lacks it), top input/output ROIs with synapse counts,
    and the top up/down-stream partner types — each with a PREDICTED synapse sign
    (excitatory/inhibitory/modulatory) from its NT. On an unknown type, returns fuzzy
    suggestions (and, for a brain-region name, cell types in that region) instead of
    failing.
    """
    return connectome.build_fingerprint(cell_type, dataset, top_k).model_dump(by_alias=True)


@mcp.tool()
def neuron_fingerprint(body_id: int, dataset: str = DATASET, top_k: int = 15) -> dict:
    """Structural fingerprint of a SINGLE neuron (bodyId): its own partners, ROIs, and
    topographic position (instance + sub-compartments). Its function is inherited from
    its type; only the position is single-cell (n=1)."""
    return connectome.build_neuron_fingerprint(body_id, dataset, top_k).model_dump(by_alias=True)


@mcp.tool()
def replicate(cell_type: str, datasets: list[str] | None = None,
              dataset: str = DATASET, top_k: int = 15) -> dict:
    """Cross-dataset replication of a type's connectivity motif (structural, no LLM).

    Does the wiring recur in other connectomes (default male-cns + banc; FlyWire when
    configured)? Returns per-dataset cell counts, the Jaccard agreement of top-K
    partner-type sets vs the base, and the partners replicating across >=2 datasets with
    side-by-side synapse weights. A motif conserved across specimens is stronger
    structural evidence."""
    return replication.replicate(cell_type, dataset, datasets, top_k).model_dump(by_alias=True)


@mcp.tool()
def hypothesize(cell_type: str, dataset: str = DATASET, top_k: int = 15,
                verify: bool = True) -> dict:
    """FULL grounded functional hypothesis for a cell type (needs GEMINI_API_KEY; slow).

    Runs the whole pipeline: connectome + literature (PubMed) + LLM synthesis + an
    anti-hallucination verification pass. Returns the functions the neuron participates
    in (each grounded in a paper id and/or a specific connectivity number), tiered
    falsifiable hypotheses, proposed experiments, caveats, and verification notes."""
    fp = connectome.build_fingerprint(cell_type, dataset, top_k)
    lit = literature.fetch_literature(fp)
    return synthesize.synthesize(fp, lit, verify=verify).model_dump(by_alias=True)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
