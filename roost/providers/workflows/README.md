# Workflow templates

A template is a ComfyUI graph in **API format** with its inputs replaced by
tokens. ComfyUI still loads it — a token is a string — so you can open one,
change it, and save it back out.

## Bringing your own

Build the workflow in ComfyUI until it makes what you want. Then
**Save (API format)** — not plain Save, which writes the editor's layout
rather than the graph — and import it:

```bash
curl -X POST localhost:8477/api/media/workflows \
  -H 'Content-Type: application/json' \
  -d "{\"name\": \"wan-t2v\", \"graph\": $(cat wan_t2v_api.json)}"
```

The response lists the tokens it found. That is the useful part: it tells you
which controls the panel will have. If `prompt` is not among them, the
importer could not find your sampler's positive conditioning and you should
put `%prompt%` into the text node yourself.

## The tokens

`prompt` `negative` `seed` `steps` `cfg` `denoise` `sampler` `scheduler`
`width` `height` `frames` `fps` `motion` `batch` `checkpoint`

A token standing alone as a whole value becomes a number where a number
belongs; one inside a longer string is substituted textually, so
`"%prompt%, cinematic lighting"` works.

Only the tokens you use become controls. A template with no `%frames%` gets no
frame-count slider, which is how an image workflow and a video workflow can
live in the same directory without either offering the other's settings.

## Overriding the form

Next to `name.json`, an optional `name.roost.json`:

```json
{
  "label": "WAN 2.2 text to video",
  "kind": "video",
  "note": "14B. About four minutes for 49 frames on a 4090.",
  "params": {
    "frames": { "default": 49, "maximum": 121,
                "help": "This model was trained at 49. Above 81 it drifts." },
    "cfg":    { "default": 5.0 }
  }
}
```

Only labels, ranges, defaults and help text — a sidecar cannot add a token the
graph does not use, because the control would have nowhere to go.

## Why none ship

A template is specific to the checkpoints and custom nodes *your* ComfyUI has
installed. One written here would name models that are not on your disk and
node classes from extensions you have not added, and it would fail at the
queue with an error about a missing key. Importing your own working graph
takes one command and produces a template that is correct by construction.
