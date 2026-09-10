# Memory

Roost can remember things about you between conversations. It is off until you
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
  Roost is a client on a machine you control — there is one user, and the
  database is a file on their disk — so at-rest recoverability is the whole
  threat, and it is the half of this that is load-bearing.
* Vectors belonging to two different user ids cannot be meaningfully compared.
  Cross-user leakage is not merely forbidden by a `WHERE` clause; it is
  meaningless. True, and not currently doing any work: Roost has one user by
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
