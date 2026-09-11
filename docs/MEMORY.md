# Memory

openmirror can remember things about you between conversations. It is off until you
switch it on, and switching it on is two decisions rather than one.

## The two switches

**Remember things between conversations.** Off by default. Off means nothing is
written — not written-and-ignored, not written-pending-consent. The text never
even reaches the embedding model. A toggle that only stopped the *reading*
would leave a record of someone who believed they had opted out.

**Also pick things up automatically.** A separate switch, because passively
learning from what you say is a bigger step than storing what you explicitly
asked to be stored. It cannot outlive the first switch: turning memory off
turns this off too, rather than leaving it armed for whenever memory is next
enabled.

Turning memory off stops storing and stops recalling. It does *not* delete what
is already there — and the UI says so, with the count, rather than implying the
data is gone.

## How it is stored

SQLite, and a full scan. A person's memories number in the thousands, not the
millions; at that size an approximate index buys nothing measurable and costs a
service to run and a second thing that can be down. A quantised vector is 768
bytes, ten thousand of them is seven megabytes, and a dot product over seven
megabytes takes a couple of milliseconds. When someone has a million memories
this will be the wrong choice, it will be obvious, and the interface will not
have to change.

Vectors are quantised to int8 with a per-vector scale: a quarter of the size,
for a similarity error of about 0.002 — well below the noise floor of the
embedding itself.

## The per-user key

Each user gets a random key, and their vectors are scrambled with it before
storage: a permutation and a sign flip.

Both are **orthogonal transforms**, so dot products — and therefore cosine
similarity — are preserved *exactly*. There is no accuracy cost at all. The
test suite asserts this to 1e-6, because the whole design rests on it.

What it buys:

* **A stolen database is not a pile of embeddings** someone can run through a
  public model to recover the text. This is the one that earns the design.
  openmirror is a client on a machine you control — there is one user, and the
  database is a file on their disk — so at-rest recoverability is the whole
  threat, and it is the half of this that is load-bearing.
* Vectors belonging to two different user ids cannot be meaningfully compared.
  Cross-user leakage is not merely forbidden by a `WHERE` clause; it is
  meaningless. True, and not currently doing any work: openmirror has one user by
  design and is not heading for accounts. Kept because the store is keyed that
  way and removing the keying to simplify would cost the bullet above.

Queries are projected the same way, so nothing is ever unprojected and the key
never has to leave its table. A wipe deletes the key along with the rows, so a
restored backup cannot be searched either.

## What gets remembered automatically

Only sentences that read like a durable fact about you or your setup — "I
always run the tests before pushing", "my deploy script lives in bin/release".
At most three per message, so one conversation cannot fill memory with a single
afternoon's opinion.

Anything transient is skipped: "today", "right now", "currently". A fact that
was only true this afternoon is a liability next week, when it will be recalled
as though it were still current.

## What it looks like to the model

Relevant memories are put in front of the model before the first request of a
turn, framed as background rather than instruction — it is what you told it
before, not what you are asking for now. There are also `remember` and `recall`
tools, because automatic recall only ever sees the opening message: an agent
that reads three files and *then* realises it needs to know how you like
migrations written has no other way to ask.

`remember` is graded as a **write**, not a read. Storing something in your
long-term memory is a side effect, and you get to say no to it like any other.

The tools are only offered when memory is on for that person. A model shown a
`remember` tool it will always be refused for using wastes a step every turn.

## Failing safely

If the embedding provider is down, recall returns nothing and the turn
continues. An assistant that refuses to answer because it could not remember is
worse than one that simply does not remember.

## Changing the embedding model

A vector only means anything relative to the model that made it. Change the
embedding model and every stored vector is a coordinate in a space that no
longer exists — and the failure is silent, which is the whole problem. Old
vectors do not *fail* against a new query; they score. Cosine against unrelated
geometry returns plausible middling numbers, so an unreset store keeps answering
questions, just with whichever old memory happened to land near the new query.
Nothing logs and nothing raises.

So openmirror notices instead. The store records which embedder its vectors are
under, and compares it at every open and before every embedding — the embedding
route can be re-pointed while openmirror is running, so a start-up-only check would
leave the rest of that session comparing against the old model. When it has
moved, **every vector is discarded and every text is kept**, and the memories are
embedded again in the background. Recall is briefly empty and says so, rather
than being briefly wrong and saying nothing.

What counts as "the same embedder" includes the width: `text-embedding-3-large`
asked for 512 dimensions and the same model at its full width are two different
embedders, because their vectors are no more comparable than two different
models' would be. The stored width (`src_dim`) already separated models of
different sizes; what it could not catch is two *different* models of the *same*
size, which is the case this exists for.

## Controlling the index by hand

The vector index is derived data: the memory is the text, and the vectors and
the search index are both things openmirror can make again. Everything here is
therefore safe to reach for, which `DELETE /api/memory` — the wipe — is not.

| | |
|---|---|
| `GET /api/memory/index` | Which embedder, how many memories, how many searchable, how many still pending, and whether `sqlite-vec` is answering or the scan is. |
| `POST /api/memory/index/rebuild` | Rebuild the search index from the vectors already stored. Local, no provider call. For an index that was interrupted, or never built because the extension arrived later. |
| `POST /api/memory/index/reset` | Throw away every vector and make them again from the texts. The manual form of what a changed model does automatically. Pass `{"reembed": false}` to do the wipe alone. |
| `POST /api/memory/index/reembed` | Embed whatever is still pending. Safe to call repeatedly. |

A re-embed is batched and stops at the first failed batch rather than hammering
a provider that is down. The rows it did not reach stay pending, so the next
call — or the next start-up — resumes exactly where it left off.
