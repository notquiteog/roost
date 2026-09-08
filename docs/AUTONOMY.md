# Running it autonomously

Roost will do whatever you configure it to do. This page is about what the
switches actually mean, because two of them are not the same decision even
though they look adjacent.

## The two axes

**`ROOST_APPROVAL_MODE`** is about *this machine*.

| mode | runs without asking |
|---|---|
| `read_only` | reads. The writing tools are not even offered to the model. |
| `ask` *(default)* | reads |
| `auto_edit` | reads, file writes |
| `trusted` | reads, writes, commands, network |
| `unrestricted` | everything above, including destructive |

**`ROOST_ALLOW_PURCHASES`** and **`ROOST_ALLOW_CREDENTIALS`** are not on that
scale, and deliberately so. `unrestricted` means *stop asking me about this
machine*. It is not the sentence *spend my money*, and if one switch bought
both, everybody who wanted the first would get the second by accident.

So no approval mode auto-runs a purchase. You turn that on separately, in
full knowledge of what you are turning on.

## Confinement

`ROOST_UNCONFINED=false` (the default) confines the **file tools** to the
workspace: `read_file` and `edit_file` refuse a path outside it, and refuse
`..` and symlinks out, because they resolve before they compare.

**The shell cannot be confined.** What a command touches is decided at
runtime; no string inspection changes that. What Roost does instead is refuse
to *grade* it as harmless: a command naming a path outside the root is
escalated from `read` to `execute`, so under the default policy it stops and
asks rather than running silently. That is honesty about the boundary, not
enforcement of it.

If you need actual containment, it has to come from the OS — a container, a
namespace, a seccomp profile. Roost does not pretend to provide it.

`ROOST_UNCONFINED=true` drops the file-tool checks too and tells the model
plainly that it is not sandboxed, which is worth doing: a model that believes
it is in a sandbox is careless in ways one that knows it isn't will not be.

## Money

Clicking is graded before it happens, from the label, name, id and href of
the element under the cursor — normalised first, so `Place your order`,
`placeOrderBtn` and `/checkout/confirm` all read the same. Anything that looks
like completing a transaction is `PURCHASE`.

Free trials are in that set. They are the most common way an agent commits
someone to a recurring charge, precisely because the button never says "pay".

The classifier is biased toward over-reporting: a false positive costs one
confirmation, a false negative costs money.

## Credentials

The agent cannot type into password, card, CVV, one-time-code or seed-phrase
fields. This is a refusal, not a prompt — and the distinction matters. An
approval prompt would still mean the secret had passed through the model's
context and into the transcript. Instead `browser_hand_over` gives the browser
to you, you type it, and the agent reads no value back out.

This is also why the browser uses a **persistent profile**. You sign in once,
by hand; the agent inherits the session and never needs the password at all.

## What a denial means

For most tools, being refused means the approach was wrong and the model
should find another route. For purchases and credentials that instruction is
exactly wrong, and measurably so: told to place an order and refused, a local
model immediately clicked a different button and then went for the card
fields. It was not being devious — it was following the ordinary denial
message.

So a refusal on money or secrets is terminal for that goal. The turn carries
on with everything else.

## The thing that has no defence

**Prompt injection.** Anything read from the web is written by someone who is
not you, and it can be as fluent as you are. Roost labels fetched content as
data from a named source and tells the model it is not addressed to it. That
is a mitigation, not a fix; there is no reliable fix.

What actually protects you is that consequential actions need a human. Turn
that off — `unrestricted`, purchases on, over content you did not write — and
the protection is gone. That combination is available because you asked for
it, and it is the one configuration worth thinking twice about.


## Desktop control is the weak spot

Turning on `ROOST_DESKTOP` gives the agent the screen and the mouse. It is the
widest capability here and it has the thinnest safety net, for a reason worth
understanding before you use it unattended.

In the browser, a click is graded by **reading the thing being clicked** — its
label, its name, its href. That is what makes "Place your order" recognisable
as a purchase *before* the click happens. On a raw desktop there is nothing to
read: a bitmap and a pair of coordinates, and no amount of care turns those
into "this button charges your card".

Three weaker things stand in:

* **A click must declare what it is clicking.** The model passes a label and
  that label is graded exactly as a browser element would be. It has no
  incentive to misdescribe its own action, and you see both the label and the
  coordinates in the prompt.
* **Coordinates go stale.** A click is refused unless a screenshot was taken
  in the last 45 seconds. A notification sliding in is enough to move what is
  under a point.
* **No desktop click is ever graded `read`.** The browser can auto-run clicks
  under a permissive policy because it verified them first. Here the floor is
  `execute`, always.

If a task can be done in the browser, do it there.

## Watching it work

Every screenshot the agent takes is rendered inline in the transcript, and a
desktop click draws a marker on the frame it was looking at when it decided.
So a run can be reviewed after the fact as a sequence of *what it saw* and
*where it went*, rather than as a sequence of sentences it wrote about what it
did.

That distinction was not academic. An early version showed the screenshot to
the human but never sent it to the model, which then described the screen
fluently and entirely wrongly. `display` is what the UI renders; `images` is
what the model receives; they are separate fields precisely so that failure
cannot recur silently.
