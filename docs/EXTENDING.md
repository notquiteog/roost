# Extending it

Two mechanisms, and one thing that undoes what they do.

## MCP servers

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
    "db":     { "command": "uvx", "args": ["mcp-db"], "trustHints": true }
  }
}
```

Tools arrive as `mcp__<server>__<tool>`.

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
