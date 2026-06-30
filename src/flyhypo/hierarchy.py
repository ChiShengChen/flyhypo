"""Resolve the analysis hierarchy for a query and build the structural context.

Levels (fine → coarse): neuron → cell_type → umbrella → subregion → region.
- neuron     : one bodyId (optional; only when --neuron is given)
- cell_type  : the neuPrint type (e.g. EPG)
- umbrella   : the functional SYSTEM / type-family — NAMED by the LLM (e.g.
               "central-complex compass system"); structure summarised from the type
- subregion  : the type's dominant sub-compartment ROI (e.g. EBr2r4)
- region     : the type's dominant primary neuropil ROI (e.g. EB)

This module only assembles the structural evidence; synthesize_hierarchy turns it
into per-level functional roles.
"""

from __future__ import annotations

from . import connectome
from .schema import StructuralFingerprint


def _primary_region(fp: StructuralFingerprint) -> str | None:
    if fp.input_rois:
        return fp.input_rois[0].roi
    if fp.output_rois:
        return fp.output_rois[0].roi
    return None


def _trim(fp: StructuralFingerprint) -> dict:
    return {
        "predicted_nt": fp.predicted_nt,
        "input_rois": [(r.roi, r.weight) for r in fp.input_rois],
        "output_rois": [(r.roi, r.weight) for r in fp.output_rois],
        "upstream": [(p.type, p.n_cells, p.total_weight) for p in fp.upstream[:8]],
        "downstream": [(p.type, p.n_cells, p.total_weight) for p in fp.downstream[:8]],
        "sub_rois": [(r.roi, r.weight) for r in fp.sub_rois],
    }


def build_context(
    query: str | None,
    dataset: str,
    top_k: int,
    body_id: int | None = None,
    *,
    use_cache: bool = True,
):
    """Return (context_dict, type_fp, neuron_fp). type_fp may be None if unresolved."""
    neuron_fp = None
    if body_id is not None:
        neuron_fp = connectome.build_neuron_fingerprint(
            body_id, dataset, top_k, use_cache=use_cache
        )
        ntype = neuron_fp.neuron_type
    else:
        ntype = query

    type_fp = None
    if ntype:
        type_fp = connectome.build_fingerprint(ntype, dataset, top_k, use_cache=use_cache)

    region = _primary_region(type_fp) if type_fp and type_fp.found else None
    subregion = None
    if neuron_fp and neuron_fp.sub_rois:
        subregion = neuron_fp.sub_rois[0].roi
    elif type_fp and type_fp.sub_rois:
        subregion = type_fp.sub_rois[0].roi

    client = connectome.get_client(dataset)
    region_types = connectome.region_dominant_types(client, region) if region else []
    sub_types = connectome.region_dominant_types(client, subregion) if subregion else []

    context = {
        "query": query if body_id is None else f"bodyId:{body_id}",
        "dataset": dataset,
        "cell_type": ntype,
        "region": region,
        "subregion": subregion,
        "neuron_bodyId": body_id,
        "type_fingerprint": _trim(type_fp) if (type_fp and type_fp.found) else None,
        "neuron": (
            {
                "bodyId": body_id,
                "instance": neuron_fp.neuron_instance,
                "type": neuron_fp.neuron_type,
                "n_in_type": neuron_fp.n_in_type,
                "sub_rois": [(r.roi, r.weight) for r in neuron_fp.sub_rois],
            }
            if neuron_fp and neuron_fp.found
            else None
        ),
        "region_dominant_types": region_types,
        "subregion_dominant_types": sub_types,
    }
    return context, type_fp, neuron_fp
