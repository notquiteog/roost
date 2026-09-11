/* The companion runner.
 *
 * A companion is a small pixel creature that lives at the bottom of the
 * transcript and reacts to what the agent is doing — reading, running a
 * command, waiting on you, failing. It is decoration, but not only: the same
 * information is in the tool cards, and a glance at the corner is cheaper
 * than reading them.
 *
 * Nothing in here knows about any particular creature. A companion is data —
 * a palette, a set of frames written as strings of characters, and a table of
 * states — and adding one is adding a file next to this one. The runner
 * cycles frames, moves the sprite around the stage, and picks which state to
 * render; everything expressive lives in the creature's own file.
 *
 * The state vocabulary is shared, so the rest of the client never learns
 * which companion is on. A creature that does not define a state falls back
 * along CHAIN, which is what lets one define three states and another
 * fifteen and both still work.
 */

/* The stage is measured in sprite pixels, not CSS ones: everything positional
   here is on the pixel grid, and the only place a real pixel appears is the
   transform set up in `resize`. */
export const STAGE = { w: 36, h: 22, scale: 4 };

/* How long a piece of work has to run before the creature starts to fidget.
   Short enough to catch a slow build, long enough that an ordinary tool call
   never triggers it. */
const GRIND_AFTER = 25;

/* What to try when a creature has not defined the state it was asked for.
   Written out per state rather than as a parent pointer so that a chain can
   never loop, which a table of fallbacks assembled at runtime very easily
   can. */
const CHAIN = {
  idle: ['idle'],
  typing: ['typing', 'idle'],
  thinking: ['thinking', 'work', 'idle'],
  reading: ['reading', 'work', 'thinking', 'idle'],
  writing: ['writing', 'work', 'thinking', 'idle'],
  running: ['running', 'work', 'thinking', 'idle'],
  browsing: ['browsing', 'work', 'thinking', 'idle'],
  desktop: ['desktop', 'browsing', 'work', 'thinking', 'idle'],
  grinding: ['grinding', 'work', 'thinking', 'idle'],
  waiting: ['waiting', 'idle'],
  listening: ['listening', 'waiting', 'idle'],
  speaking: ['speaking', 'thinking', 'idle'],
  success: ['success', 'idle'],
  error: ['error', 'idle'],
};

/* The states that mean "it is working on something", and so the ones that can
   turn into `grinding` if they go on long enough. */
const WORKING = new Set(['thinking', 'reading', 'writing', 'running', 'browsing', 'desktop']);

/* One tool at a time is running, and what it is doing is more interesting
   than the fact that a tool is running at all. Names come from the server's
   tool registry; anything unrecognised is treated as work rather than
   dropped, so a tool added later still moves the creature. */
const TOOL_STATES = {
  read_file: 'reading',
  read_files: 'reading',
  outline: 'reading',
  plan: 'thinking',
  list_dir: 'reading',
  glob: 'reading',
  grep: 'reading',
  recall: 'reading',
  write_file: 'writing',
  edit_file: 'writing',
  multi_edit: 'writing',
  apply_patch: 'writing',
  remember: 'writing',
  shell: 'running',
  web_search: 'browsing',
  web_fetch: 'browsing',
  research: 'browsing',
  // An MCP resource is somebody else's document over a socket. That is reading
  // from where the person is sitting, whatever the wire underneath is.
  mcp_list_resources: 'reading',
  mcp_read_resource: 'reading',
  browser_navigate: 'browsing',
  browser_read: 'browsing',
  browser_click: 'browsing',
  browser_type: 'browsing',
  browser_screenshot: 'browsing',
  browser_hand_over: 'waiting',
  desktop_screenshot: 'desktop',
  desktop_click: 'desktop',
  desktop_type: 'desktop',
  desktop_key: 'desktop',
  desktop_scroll: 'desktop',
  // Generation is a long wait on somebody else's GPU, which is what
  // 'running' already means here. It gets its own word in the caption rather
  // than its own animation: a sixth state would need frames drawing in five
  // creatures, and the wait looks the same from outside either way.
  generate_image: 'running',
  generate_video: 'running',
  system_info: 'reading',
  display_info: 'reading',
  package_search: 'browsing',
  // Installing is a long wait on somebody else's servers and then a lot of
  // unpacking, which is what 'running' already means here.
  package_install: 'running',
  package_remove: 'running',
  display_hdr: 'desktop',
  media_params: 'reading',
  media_job: 'reading',
  import_workflow: 'writing',
  ask_user: 'waiting',
};

export function stateForTool(name) {
  return TOOL_STATES[name] || 'running';
}

/* ------------------------------------------------------------- validation */

/* Pixel art written as strings has exactly one failure mode: a row that is a
   character short, which shifts everything after it and is invisible in a
   diff. Checking costs a few hundred string comparisons at selection time and
   turns that into a named error. */
export function problems(def) {
  const found = [];
  const say = (msg) => found.push(`${def.id}: ${msg}`);

  for (const [name, rows] of Object.entries(def.frames)) {
    if (!Array.isArray(rows) || !rows.length) {
      say(`frame ${name} has no rows`);
      continue;
    }
    const width = rows[0].length;
    rows.forEach((row, y) => {
      if (row.length !== width) say(`frame ${name} row ${y} is ${row.length} wide, expected ${width}`);
      for (const ch of row) {
        if (ch !== '.' && !(ch in def.palette)) say(`frame ${name} row ${y} uses "${ch}", which is not in the palette`);
      }
    });
  }

  for (const [name, state] of Object.entries(def.states)) {
    for (const frame of state.frames || []) {
      if (!(frame in def.frames)) say(`state ${name} names frame ${frame}, which does not exist`);
    }
  }
  if (!def.states.idle) say('has no idle state');
  return found;
}

/* ---------------------------------------------------------------- painting */

/* Frames are painted once at one sprite pixel per canvas pixel and then
   scaled up with smoothing off. Filling a few hundred rectangles per frame
   would also work, but baking keeps the draw loop to one drawImage per
   sprite, which matters on a machine already running a model. */
const oven = new Map();

function bake(def, frameName, variant) {
  const key = `${def.id}/${frameName}/${variant || ''}`;
  const cached = oven.get(key);
  if (cached !== undefined) return cached;

  const rows = def.frames[frameName];
  if (!rows) {
    oven.set(key, null);
    return null;
  }
  const palette = variant && def.variants && def.variants[variant]
    ? { ...def.palette, ...def.variants[variant] }
    : def.palette;

  const canvas = document.createElement('canvas');
  canvas.width = rows[0].length;
  canvas.height = rows.length;
  const ctx = canvas.getContext('2d');
  for (let y = 0; y < rows.length; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const colour = palette[row[x]];
      if (!colour) continue;               // '.' and anything unpainted
      ctx.fillStyle = colour;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  oven.set(key, canvas);
  return canvas;
}

/* A still portrait for the picker: the creature as it looks when nothing is
   happening, plus whatever it is normally holding or standing in. Declared by
   the companion rather than guessed, because half of them look like nothing at
   all in their idle frame — the mandrake's is two leaves. */
export function portrait(def, scale) {
  const spec = def.portrait || { frame: def.states.idle.frames[0] };
  const base = bake(def, spec.frame, null);
  if (!base) return null;
  const canvas = document.createElement('canvas');
  canvas.width = base.width * scale;
  canvas.height = base.height * scale;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.drawImage(base, 0, 0);
  for (const [name, x, y] of spec.props || []) {
    const sprite = bake(def, name, null);
    if (sprite) ctx.drawImage(sprite, x, y);
  }
  canvas.style.width = `${canvas.width}px`;
  canvas.style.height = `${canvas.height}px`;
  return canvas;
}

/* ------------------------------------------------------------- the runner */

const reduced = typeof window !== 'undefined' && window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : null;

export class Perch {
  /* `scale` is how many canvas pixels one sprite pixel gets. It is per-perch
     rather than global because the same creature now appears at several
     sizes at once — a hand-sized one on the perch, a thumbnail in the
     composer, a big one on an empty session — and they all animate off the
     same state. */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scale = opts.scale || STAGE.scale;
    this.def = null;
    this.ambient = 'idle';
    this.since = performance.now();
    this.flashing = null;
    this.flashFrom = 0;
    this.flashUntil = 0;
    this.frame = null;
    this.timer = null;
    this.resize();
    // Kept on the instance so `destroy` can take them off again: perches are
    // created and thrown away with the cards they live on now, and a listener
    // per dead perch is a leak with a frame loop attached.
    this.onResize = () => this.resize();
    this.onWake = () => this.pump();
    window.addEventListener('resize', this.onResize);
    // A creature animating in a tab nobody is looking at is pure waste, and
    // on a laptop it is waste with a battery attached.
    document.addEventListener('visibilitychange', this.onWake);
    // Turned on or off while the page is open, not only at load.
    if (reduced && reduced.addEventListener) reduced.addEventListener('change', this.onWake);
  }

  /* Take a perch down for good. Everything transient in the client — an
     approval bar, a voice strip — can carry a creature, so they have to be
     able to stop being one. */
  destroy() {
    this.def = null;
    this.pump();
    window.removeEventListener('resize', this.onResize);
    document.removeEventListener('visibilitychange', this.onWake);
    if (reduced && reduced.removeEventListener) reduced.removeEventListener('change', this.onWake);
  }

  /* Motion is decoration, and decoration that cannot be turned off is a
     problem for anyone who gets sick from it. */
  frozen() {
    return Boolean(reduced && reduced.matches);
  }

  resize() {
    // Whole device pixels only. A fractional scale on nearest-neighbour art
    // gives some rows two pixels and others one, which reads as a wobble.
    const ratio = Math.max(1, Math.round(window.devicePixelRatio || 1));
    this.unit = this.scale * ratio;
    this.canvas.width = STAGE.w * this.unit;
    this.canvas.height = STAGE.h * this.unit;
    this.canvas.style.width = `${STAGE.w * this.scale}px`;
    this.canvas.style.height = `${STAGE.h * this.scale}px`;
    this.ctx.imageSmoothingEnabled = false;
  }

  /* Put a creature on the perch, or take the perch down with null. Returns
     the problems found in the art, if any: a broken companion is skipped
     rather than thrown, because a decoration must not be able to take the
     page with it. */
  use(def) {
    const found = def ? problems(def) : [];
    this.def = found.length ? null : def;
    this.ambient = 'idle';
    this.since = performance.now();
    this.flashing = null;
    this.clear();
    this.pump();
    return found;
  }

  /* The state the agent is in now, held until something else replaces it. */
  set(name) {
    if (!CHAIN[name] || name === this.ambient) return;
    this.ambient = name;
    this.since = performance.now();
    this.pump();
  }

  /* A state that plays through and hands back to whatever the agent is doing.
     Success and failure are moments, not conditions. */
  flash(name, ms) {
    if (!CHAIN[name]) return;
    const state = this.def && this.resolve(name);
    this.flashing = name;
    this.flashFrom = performance.now();
    this.flashUntil = this.flashFrom + (ms || (state && state.once) || 1200);
    this.pump();
  }

  tool(name) {
    this.set(stateForTool(name));
  }

  /* Typing is not a state the agent is in, so it expires on its own rather
     than needing a matching "stopped typing". */
  typing() {
    if (this.ambient !== 'idle') return;
    this.flash('typing', 700);
  }

  resolve(name) {
    for (const candidate of CHAIN[name] || ['idle']) {
      if (this.def.states[candidate]) return this.def.states[candidate];
    }
    return null;
  }

  clear() {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /* A frame loop while something is moving; a slow timer when nothing is,
     which exists only to notice that a flash has finished. Neither runs with
     the tab in the background. */
  pump() {
    const wanted = Boolean(this.def) && !document.hidden;
    const still = wanted && this.frozen();

    if (this.frame !== null && (!wanted || still)) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
    if (this.timer !== null && (!wanted || !still)) {
      clearInterval(this.timer);
      this.timer = null;
    }

    if (!wanted) {
      this.clear();
      return;
    }
    if (still) {
      this.draw(performance.now());
      if (this.timer === null) this.timer = setInterval(() => this.draw(performance.now()), 400);
    } else if (this.frame === null) {
      this.frame = requestAnimationFrame((now) => this.tick(now));
    }
  }

  tick(now) {
    this.frame = null;
    if (!this.def) return;
    this.draw(now);
    this.pump();
  }

  /* What is being rendered this instant: the flash if one is running, the
     ambient state otherwise, promoted to `grinding` when it has gone on long
     enough to be worth remarking on. */
  view(now) {
    if (this.flashing && now < this.flashUntil) {
      const span = this.flashUntil - this.flashFrom;
      return {
        state: this.flashing,
        t: (now - this.flashFrom) / 1000,
        p: Math.min(1, (now - this.flashFrom) / span),
      };
    }
    this.flashing = null;
    const t = (now - this.since) / 1000;
    const state = WORKING.has(this.ambient) && t > GRIND_AFTER ? 'grinding' : this.ambient;
    return { state, t, p: 0 };
  }

  draw(now) {
    const def = this.def;
    const ctx = this.ctx;
    const view = this.view(now);
    view.home = def.home;
    view.still = this.frozen();

    const state = this.resolve(view.state);
    if (!state) return;

    // Held still for anyone who asked not to be animated: the clock stops, so
    // nothing cycles, drifts or flaps. What the creature is *doing* still
    // changes with the agent, because that part is information rather than
    // decoration, and a frozen frame carries it just as well.
    if (view.still) {
      view.t = 0;
      view.p = 0;
    }
    const at = state.motion && !view.still ? state.motion(view) : {};
    const pos = {
      x: at.x === undefined ? def.home.x : at.x,
      y: at.y === undefined ? def.home.y : at.y,
      flip: Boolean(at.flip),
    };

    ctx.setTransform(this.unit, 0, 0, this.unit, 0, 0);
    ctx.clearRect(0, 0, STAGE.w, STAGE.h);

    for (const prop of def.props || []) {
      if (!prop.front) this.paintProp(prop, view, pos, state.variant);
    }

    const name = state.pick
      ? state.pick(view)
      : state.frames[Math.floor(view.t * (state.fps || 4)) % state.frames.length];
    if (name) this.paint(name, pos.x, pos.y, pos.flip, state.variant);

    for (const prop of def.props || []) {
      if (prop.front) this.paintProp(prop, view, pos, state.variant);
    }
  }

  /* Props take the colour the creature is currently wearing unless they pin
     one of their own: a screen that stays green while the case flashes red
     would read as two separate things. */
  paintProp(prop, view, pos, variant) {
    const name = prop.pick(view);
    if (!name) return;
    const x = prop.attach ? pos.x + prop.x : prop.x;
    const y = prop.attach ? pos.y + prop.y : prop.y;
    this.paint(name, x, y, false, prop.variant || variant);
  }

  paint(name, x, y, flip, variant) {
    const sprite = bake(this.def, name, variant);
    if (!sprite) return;
    // Snapped to whole device pixels: motion that lands between them turns
    // crisp art into a smear, and the grid is the whole point of the style.
    const px = Math.round(x * this.unit) / this.unit;
    const py = Math.round(y * this.unit) / this.unit;
    const ctx = this.ctx;
    if (flip) {
      ctx.save();
      ctx.translate(px + sprite.width, py);
      ctx.scale(-1, 1);
      ctx.drawImage(sprite, 0, 0);
      ctx.restore();
    } else {
      ctx.drawImage(sprite, px, py);
    }
  }
}
