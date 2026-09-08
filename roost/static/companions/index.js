/* The companion picker, and the one object the rest of the client talks to.
 *
 * Which creature is on is a preference, not a setting: it is kept in this
 * browser, never sent anywhere, and turning it off leaves nothing behind but
 * a faint mark to turn it back on with.
 */

import { Perch, portrait } from './engine.js';
import moth from './moth.js';
import gargoyle from './gargoyle.js';
import slime from './slime.js';
import mechbit from './mechbit.js';
import mandrake from './mandrake.js';

export const COMPANIONS = [moth, gargoyle, slime, mechbit, mandrake];

const STORE = 'roost.companion';
const DEFAULT = 'moth';
const NONE = 'none';

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
    this.perch = null;
    this.id = null;
    this.root = null;
    this.quietUntil = 0;
  }

  mount(root) {
    if (!root) return;
    this.root = root;
    const canvas = el('canvas');
    canvas.setAttribute('aria-hidden', 'true');

    const button = el('button', 'pet');
    button.type = 'button';
    button.setAttribute('aria-haspopup', 'true');
    button.setAttribute('aria-expanded', 'false');
    button.appendChild(canvas);
    button.appendChild(el('span', 'empty', '◌'));

    const menu = el('div', 'companion-menu');
    menu.hidden = true;

    root.appendChild(button);
    root.appendChild(menu);
    this.button = button;
    this.menu = menu;
    this.perch = new Perch(canvas);

    button.onclick = (e) => {
      e.stopPropagation();
      this.toggleMenu();
    };
    // Anywhere else, and Escape, close it — the same two ways every menu in
    // every application closes.
    document.addEventListener('click', () => this.closeMenu());
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeMenu();
    });

    this.buildMenu();
    this.choose(remembered() || DEFAULT);
  }

  buildMenu() {
    const choose = (id) => (e) => {
      e.stopPropagation();
      this.choose(id);
      this.closeMenu();
      this.button.focus();
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
      this.menu.appendChild(row);
    }

    const off = el('button', 'row off');
    off.type = 'button';
    const text = el('div', 'text');
    text.appendChild(el('div', 'name', 'Nobody'));
    text.appendChild(el('div', 'blurb', 'an empty corner'));
    off.appendChild(text);
    off.onclick = choose(NONE);
    off.dataset.id = NONE;
    this.menu.appendChild(off);
  }

  toggleMenu() {
    this.menu.hidden ? this.openMenu() : this.closeMenu();
  }

  openMenu() {
    this.menu.hidden = false;
    this.button.setAttribute('aria-expanded', 'true');
  }

  closeMenu() {
    if (!this.menu || this.menu.hidden) return;
    this.menu.hidden = true;
    this.button.setAttribute('aria-expanded', 'false');
  }

  choose(id) {
    const wanted = COMPANIONS.find((c) => c.id === id) || null;
    this.id = wanted ? wanted.id : NONE;
    remember(this.id);

    // A creature whose art does not check out is refused rather than drawn
    // wrong, so the perch has to end up looking empty even though something
    // was asked for — otherwise there is nothing left to click.
    const broken = this.perch.use(wanted);
    for (const problem of broken) console.error(`companion: ${problem}`);
    const def = broken.length ? null : wanted;

    this.root.classList.toggle('off', !def);
    this.button.title = def ? `${def.label} — click to change` : 'Companion';
    this.button.setAttribute('aria-label', def ? `Companion: ${def.label}. Choose another.` : 'Choose a companion');
    for (const row of this.menu.querySelectorAll('.row')) {
      row.setAttribute('aria-current', String(row.dataset.id === this.id));
    }
  }

  /* Everything below is what the client actually calls. Each is a no-op when
     there is no companion, so nothing that reports state has to check first. */

  set(state) {
    if (this.perch) this.perch.set(state);
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
    if (!this.perch || performance.now() < this.quietUntil) return;
    this.perch.flash(state);
  }

  tool(name) {
    if (this.perch) this.perch.tool(name);
  }

  typing() {
    if (this.perch) this.perch.typing();
  }
}

export const companion = new Companion();
