"""MCP server exposes the expected tools with usable schemas (no network)."""

import asyncio

from flyhypo.mcp_server import mcp


def test_tools_registered():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert {"fingerprint", "neuron_fingerprint", "replicate", "hypothesize"} <= set(tools)


def test_tool_schemas():
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "cell_type" in tools["fingerprint"].inputSchema["properties"]
    assert "body_id" in tools["neuron_fingerprint"].inputSchema["properties"]
    assert "verify" in tools["hypothesize"].inputSchema["properties"]
    # every tool carries a description (becomes the calling agent's tool doc)
    assert all(t.description for t in tools.values())
