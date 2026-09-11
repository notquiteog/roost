# Extending it

Three mechanisms, and one thing that undoes what they do. Two of them are MCP
pointing in opposite directions: openmirror using everybody else's tools, and
everybody else's clients using openmirror.

## Using MCP servers

openmirror is an MCP client, so any MCP server becomes tools the agent can use.
Servers are read from a `.mcp.json` in the shape the rest of the ecosystem
already uses, which means a file written for another client works here
unchanged:

```json
{
  "mcpServers": {
    "files":  { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv"] },
    "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": { "GITHUB_TOKEN": "..." } },
    "db":     { "command": "uvx", "args": ["mcp-db"], "trustHints": true },

    "hosted": { "type": "http", "url": "https://example.com/mcp",
                "headers": { "Authorization": "Bearer ..." } },
    "legacy": { "url": "https://example.com/sse" }
  }
}
```

Tools arrive as `mcp__<server>__<tool>`.

### Local and remote

A server is a `command` to run or a `url` to reach. Remote ones speak one of
two HTTP transports: the current streamable one (`"type": "http"`) and the
older SSE pair (`"type": "sse"`). A URL ending in `/sse` is assumed to be the
older one, since that is the path every implementation of it uses; say `type`
explicitly to override the guess. `headers` are sent with every request, which
is where a hosted server's token goes — per server, because two servers needing
two different tokens is the normal case.

A remote server is the case where *not* running somebody else's code on your
machine is the point. It is also the case where the untrusted-server floor
below matters most: a hosted tool can change what it does between one call and
the next, and nothing local will have noticed.

### Resources, not just tools

A server can also expose **resources** — documents, records, pages, addressed
by URI — and **prompts**. Resources arrive as a fixed pair of tools,
`mcp_list_resources` and `mcp_read_resource`, rather than one tool each: a
server with four hundred documents would otherwise put four hundred entries in
the model's tool list, which is the failure `TOOLSETS` exists to prevent
arriving from outside the project.

The pair only appears when a connected server actually declares resources. A
model shown `mcp_read_resource` by an install that has none will try it, be
told there is nothing, and have spent a turn learning what the tool list could
have told it by not being there.

Reading a resource is graded `network` rather than `read` — the same as
`web_fetch`, and for the same reason. What comes back is somebody else's
document, fetched from somewhere else again, and the risk is what arrives
rather than what it does.

### Why `trustHints` exists and defaults to false

MCP lets a tool declare `readOnlyHint: true`. Taking that at face value would
let any server exempt itself from approval — including one installed by pasting
a command off a web page, which is how most of them get installed. So hints are
ignored unless you mark that server trusted, and until then **every MCP tool is
at least `execute`**.

The distrust is one-directional. A `destructiveHint` is believed from any
server: volunteering that something is dangerous is not a claim worth doubting.

Names and descriptions are graded regardless of hints, so a tool called
`create_order` reads as a purchase whatever it says about itself. That check
runs over text the server controls, which sounds like a weakness and is not —
it can only *raise* the risk, so a hostile description makes a tool harder to
run, never easier.

A server's descriptions and its results both go into the model's context, so
both are labelled as coming from that server and as data rather than
instructions. That is a mitigation, not a fix; see the note on prompt
injection in [AUTONOMY.md](AUTONOMY.md).

### Lifetime

Each server is a child process. One failing to start does not stop the others —
an agent with four working tool servers and one broken one is far more useful
than one that refused to start — and all of them are reaped when the daemon
stops.

## openmirror as an MCP server

The other direction: your editor, a desktop assistant or another agent using
*this* twin — its memory, its provider routing, its search backend. Without it,
the memory openmirror has been accumulating is reachable only from openmirror's own
UI.

Two transports. On stdio, for a client that would rather spawn it:

```json
{
  "mcpServers": {
    "openmirror": { "command": "openmirror-mcp" }
  }
}
```

Or over HTTP, from the daemon that is already running:

```
OPENMIRROR_MCP_SERVE=true
OPENMIRROR_MCP_SERVE_TOKEN=a-long-random-string
OPENMIRROR_MCP_SERVE_SCOPE=read
```

which serves `POST /mcp` (streamable HTTP) and `GET /mcp/sse` + `POST
/mcp/messages` (the older transport). Both are off by default.

### Scope

One setting with three values, because the decision a person actually makes is
how much of themselves to lend out, not eleven tool-shaped decisions.

| Scope | What it adds |
|---|---|
| `read` | `recall`, `web_search`, `web_fetch`, `research`, and the resources |
| `write` | `remember` |
| `all` | `generate_image` — **this spends money** |

An unrecognised scope is `read`, and says so in the log. A typo in a config
file must not be a way to widen what is exposed.

**Nothing that acts on the machine is exposed at any scope.** No shell, no
files, no desktop, no browser driving. Those are the tools openmirror guards with
an approval prompt in front of a human, and an MCP call has no human in front
of it — exposing them here would route around the one control that matters. The
approval policy is not reimplemented on the server side; the dangerous tools
are simply not there, and a test asserts it.

### The token is not optional on a network interface

This endpoint can hand out a person's memory, so on any bind that is not
loopback the router refuses to mount without `OPENMIRROR_MCP_SERVE_TOKEN`. That
is stricter than the rest of the server, which warns and carries on. The
asymmetry is deliberate: an open agent endpoint is a machine somebody else can
use, and an open twin endpoint is a person somebody else can read.

Resources it offers: `openmirror://capabilities` (what this install can do, and
what it deliberately does not expose), `openmirror://providers`,
`openmirror://memory/recent`, and the templates `openmirror://memory/search/{query}`
and `openmirror://media/{id}`.

## Rewind

Every turn that changes a file opens a checkpoint first, so a turn can be
undone:

```
GET  /api/sessions/{id}/checkpoints
POST /api/sessions/{id}/restore   {"checkpoint": "1"}
```

or the **Rewind** button in the header.

Undoing a turn undoes every turn after it too. Undoing one in the middle would
produce a tree that never existed and that nobody asked for.

Snapshots are content-addressed, so a session that edits one file thirty times
stores thirty hashes and a handful of blobs. They live under the data
directory, never inside the working root — a snapshot in the tree the agent is
editing gets read, grepped, and eventually committed.

An **interrupted** turn keeps its checkpoint. That is the turn most likely to
have left the tree somewhere nobody wanted.

If somebody edited a file by hand after the agent wrote it, the rewind says so
rather than silently overwriting their work. That comparison is against what
the agent *left*, not what it found — comparing against the before-state makes
the warning fire on every single rewind, since the agent changing the file is
the reason there is a checkpoint at all, and a warning that always fires is one
nobody reads.

### What it does not cover

**Only changes made through the file tools are recorded.** A shell command can
write anywhere, and snapshotting the whole filesystem before every command is
not something anyone wants. This is an undo for the agent's own edits, not a
transaction over your machine.

Where the working root is a git repository, git remains the better answer, and
the UI says so.
