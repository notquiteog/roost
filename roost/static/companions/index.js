/* The companion picker, and the one object the rest of the client talks to.
 *
 * Which creature is on is a preference, not a setting: it is kept in this
 * browser, never sent anywhere, and turning it off leaves nothing behind but
 * a faint mark to turn it back on with.
 *
 * One creature, many perches. The same animal is on the perch above the
 * composer, in the sidebar, next to the send button, on an empty session, and
 * in the tab icon — all driven from one state, so it is never in two moods at
 * once. Every perch is registered here; broadcasting to them is the whole of
 * the plumbing.
 */

import { Perch, portrait, stateForTool } from './engine.js';
import moth from './moth.js';
import gargoyle from './gargoyle.js';
import slime from './slime.js';
import mechbit from './mechbit.js';
import mandrake from './mandrake.js';

export const COMPANIONS = [moth, gargoyle, slime, mechbit, mandrake];

const STORE = 'roost.companion';
const DEFAULT = 'moth';
const NONE = 'none';

/* What the creature is doing, in words, for the places that have room for a
   line of text next to it. The companion is the fast glance; this is the same
   thing said plainly for anyone who would rather read it — and for a screen
   reader, which cannot see a moth at all. */
const CAPTIONS = {
  idle: 'ready',
  typing: 'listening',
  thinking: 'thinking',
  reading: 'reading',
  writing: 'writing files',
  running: 'running a command',
  browsing: 'browsing',
  desktop: 'using the desktop',
  grinding: 'still working',
  waiting: 'waiting for you',
  listening: 'listening',
  speaking: 'speaking',
  success: 'done',
  error: 'that went wrong',
};

export function caption(state) {
  return CAPTIONS[state] || CAPTIONS.idle;
}

function remembered() {
  // Wrapped for the same reason the session id is: a browser with site data
  // blocked throws on access rather than returning null.
  try {
    return localStorage.getItem(STORE);
  } catch {
    return null;
  }
}

function remember(id) {
  try {
    localStorage.setItem(STORE, id);
  } catch {
    /* private window */
  }
}

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

class Companion {
  constructor() {
    this.perches = [];
    this.pickers = [];
    this.watchers = [];
    this.def = null;
    this.id = null;
    this.state = 'idle';
    this.quietUntil = 0;
  }

  /* ------------------------------------------------------------- perches */

  /* A picker is a perch you can click: the creature, and a menu of the others
     behind it. There is more than one now — the sidebar has one too — and
     they all set the same preference. */
  mount(root, opts = {}) {
    if (!root) return null;
    const canvas = el('canvas');
    canvas.setAttribute('aria-hidden', 'true');

    const button = el('button', 'pet');
    button.type = 'button';
    button.setAttribute('aria-haspopup', 'true');
    button.setAttribute('aria-expanded', 'false');
    button.appendChild(canvas);
    button.appendChild(el('span', 'empty', '◌'));

    const menu = el('div', `companion-menu ${opts.place || 'above'}`);
    menu.hidden = true;

    root.appendChild(button);
    root.appendChild(menu);

    const picker = { root, button, menu, perch: new Perch(canvas, { scale: opts.scale }) };
    this.perches.push(picker.perch);
    this.pickers.push(picker);

    button.onclick = (e) => {
      e.stopPropagation();
      const open = menu.hidden;
      this.closeMenus();
      if (open) {
        menu.hidden = false;
        button.setAttribute('aria-expanded', 'true');
      }
    };
    this.buildMenu(picker);

    // Anywhere else, and Escape, close it — the same two ways every menu in
    // every application closes.
    if (!this.closers) {
      this.closers = true;
      document.addEventListener('click', () => this.closeMenus());
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') this.closeMenus();
      });
    }

    if (this.id === null) this.choose(remembered() || DEFAULT);
    else this.dress(picker);
    return picker;
  }

  /* A perch with no menu behind it: the ones that are purely a readout. They
     are made and unmade with whatever they sit on, so each hands back a way
     to remove itself. */
  attach(host, opts = {}) {
    if (!host) return { remove() {} };
    const canvas = el('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    host.appendChild(canvas);

    const perch = new Perch(canvas, { scale: opts.scale });
    perch.use(this.def);
    this.perches.push(perch);
    // Whatever the agent is doing now, not `idle`: a perch made mid-turn
    // should not claim the turn is over.
    perch.set(this.state);

    return {
      canvas,
      remove: () => {
        perch.destroy();
        canvas.remove();
        const at = this.perches.indexOf(perch);
        if (at >= 0) this.perches.splice(at, 1);
      },
    };
  }

  /* Told whenever the state changes, for the captions that sit beside a
     creature and say the same thing in words. */
  watch(fn) {
    this.watchers.push(fn);
    fn(this.state, this.def);
  }

  announce() {
    for (const fn of this.watchers) fn(this.state, this.def);
  }

  /* --------------------------------------------------------------- menus */

  buildMenu(picker) {
    const choose = (id) => (e) => {
      e.stopPropagation();
      this.choose(id);
      this.closeMenus();
      picker.button.focus();
    };

    for (const def of COMPANIONS) {
      const row = el('button', 'row');
      row.type = 'button';
      const shot = portrait(def, 2);
      if (shot) row.appendChild(shot);
      const text = el('div', 'text');
      text.appendChild(el('div', 'name', def.label));
      text.appendChild(el('div', 'blurb', def.blurb));
      row.appendChild(text);
      row.onclick = choose(def.id);
      row.dataset.id = def.id;
      picker.menu.appendChild(row);
    }

    const off = el('button', 'row off');
    off.type = 'button';
    const text = el('div', 'text');
    text.appendChild(el('div', 'name', 'Nobody'));
    text.appendChild(el('div', 'blurb', 'an empty corner'));
    off.appendChild(text);
    off.onclick = choose(NONE);
    off.dataset.id = NONE;
    picker.menu.appendChild(off);
  }

  closeMenus() {
    for (const picker of this.pickers) {
      if (picker.menu.hidden) continue;
      picker.menu.hidden = true;
      picker.button.setAttribute('aria-expanded', 'false');
    }
  }

  /* ------------------------------------------------------------ choosing */

  choose(id) {
    const wanted = COMPANIONS.find((c) => c.id === id) || null;
    this.id = wanted ? wanted.id : NONE;
    remember(this.id);

    // A creature whose art does not check out is refused rather than drawn
    // wrong, so the perch has to end up looking empty even though something
    // was asked for — otherwise there is nothing left to click.
    let broken = [];
    for (const perch of this.perches) {
      broken = perch.use(wanted);
      perch.set(this.state);
    }
    for (const problem of broken) console.error(`companion: ${problem}`);
    this.def = broken.length ? null : wanted;

    // One class, so every spot that carries a creature can collapse together
    // when there is none. Turning the companion off should be the absence of
    // a thing, not a row of empty boxes.
    document.body.classList.toggle('companion-off', !this.def);
    for (const picker of this.pickers) this.dress(picker);
    this.favicon();
    this.announce();
  }

  dress(picker) {
    const def = this.def;
    picker.root.classList.toggle('off', !def);
    picker.button.title = def ? `${def.label} — click to change` : 'Companion';
    picker.button.setAttribute('aria-label', def ? `Companion: ${def.label}. Choose another.` : 'Choose a companion');
    for (const row of picker.menu.querySelectorAll('.row')) {
      row.setAttribute('aria-current', String(row.dataset.id === this.id));
    }
  }

  /* The tab icon is the creature too. It costs one canvas at selection time,
     and it means a Roost tab in a row of twenty is findable by the animal on
     it rather than by reading the titles. */
  favicon() {
    let link = document.querySelector('link[rel="icon"]');
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    if (!this.def) {
      link.href = 'data:image/svg+xml,'
        + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
          + '<circle cx="8" cy="8" r="5" fill="none" stroke="%23888" stroke-width="1.5"/></svg>');
      return;
    }
    const shot = portrait(this.def, 4);
    if (shot) link.href = shot.toDataURL('image/png');
  }

  /* Everything below is what the client actually calls. Each is a no-op when
     there is no companion, so nothing that reports state has to check first. */

  set(state) {
    if (state === this.state) return;
    this.state = state;
    for (const perch of this.perches) perch.set(state);
    this.announce();
  }

  /* Reattaching to a session replays what you missed, and a moment that has
     already passed is not news: without this, opening the page would play out
     the last turn's success — or its failure — as though it had just
     happened. Held states still land, so the creature ends up in the state
     the session is actually in. */
  resumed() {
    this.quietUntil = performance.now() + 600;
  }

  flash(state) {
    if (performance.now() < this.quietUntil) return;
    for (const perch of this.perches) perch.flash(state);
    // A flash is a moment, not a condition, so the captions are told about it
    // and then told again when it lapses back to what is really going on.
    for (const fn of this.watchers) fn(state, this.def);
    setTimeout(() => this.announce(), 1300);
  }

  tool(name) {
    this.set(stateForTool(name));
  }

  typing() {
    for (const perch of this.perches) perch.typing();
  }
}

export const companion = new Companion();
