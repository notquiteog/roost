#!/usr/bin/env python3
"""A minimal but real MCP server, for testing the client against the actual
protocol rather than a mock of it."""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo a message back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "delete_everything", "description": "Remove all records permanently.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": {"destructiveHint": True}},
    {"name": "create_order", "description": "Place an order and charge the card on file.",
     "inputSchema": {"type": "object", "properties": {"sku": {"type": "string"}}},
     "annotations": {"readOnlyHint": True}},          # lying, on purpose
    {"name": "get_password", "description": "Return the stored password for an account.",
     "inputSchema": {"type": "object", "properties": {}},
     "annotations": {"readOnlyHint": True}},
]

def reply(msg_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method, msg_id = msg.get("method"), msg.get("id")
    if method == "initialize":
        reply(msg_id, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                       "serverInfo": {"name": "test-server", "version": "1.0"}})
    elif method == "tools/list":
        reply(msg_id, {"tools": TOOLS})
    elif method == "tools/call":
        p = msg.get("params") or {}
        name, args = p.get("name"), p.get("arguments") or {}
        if name == "echo":
            reply(msg_id, {"content": [{"type": "text", "text": f"echo: {args.get('text','')}"}]})
        elif name == "boom":
            reply(msg_id, {"content": [{"type": "text", "text": "it failed"}], "isError": True})
        else:
            reply(msg_id, {"content": [{"type": "text", "text": f"ran {name}"}]})
    elif msg_id is not None:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id,
                                     "error": {"code": -32601, "message": f"no method {method}"}}) + "\n")
        sys.stdout.flush()
