/* The desk-bound mandrake.
 *
 * A root that lives in a coffee mug. Most of the time all you see is two
 * leaves over the rim; when there is work it hauls itself up far enough to
 * watch, and when the work goes on and on it climbs out and paces the desk.
 *
 * The mug is drawn in front of it, which is the whole trick: submerging is
 * not a set of frames, it is the creature sitting low enough that the mug
 * covers everything but the leaves.
 */

export default {
  id: 'mandrake',
  label: 'Mandrake',
  blurb: 'lives in a coffee mug',

  home: { x: 9, y: 6 },
  portrait: { frame: 'stand', props: [['mug', 4, 8]] },

  palette: {
    r: '#d6b083',   // root, lit
    R: '#a67c50',   // root, shaded
    l: '#6cb355',   // leaf
    L: '#3f7a35',   // leaf, underside
    e: '#2a1a10',   // eyes
    f: '#ff9d3c',   // flame
    F: '#ffe08a',   // flame, core
    m: '#e5e0d6',   // mug
    M: '#a9a296',   // mug, shaded
    c: '#4a2f1e',   // what is in the mug
  },

  frames: {
    /* Under the surface: leaves only. */
    sub_a: [
      '................',
      '................',
      '................',
      '................',
      '................',
      '......l...l.....',
      '.....ll...ll....',
      '......lllll.....',
      '........l.......',
      '................',
      '................',
      '................',
      '................',
      '................',
      '................',
      '................',
    ],
    sub_b: [
      '................',
      '................',
      '................',
      '................',
      '................',
      '.....l...l......',
      '....ll...ll.....',
      '.....lllll......',
      '.......l........',
      '................',
      '................',
      '................',
      '................',
      '................',
      '................',
      '................',
    ],

    stand: [
      '................',
      '......l..l......',
      '.....ll..ll.....',
      '....lllLLlll....',
      '.....lLrrLl.....',
      '......rrrr......',
      '.....rrrrrr.....',
      '.....reRRer.....',
      '.....rrrrrr.....',
      '.....rRrrRr.....',
      '......rrrr......',
      '.....RrrrrR.....',
      '....Rr.rr.rR....',
      '...R...rr...R...',
      '......R..R......',
      '.....RR..RR.....',
    ],
    walk_a: [
      '................',
      '......l..l......',
      '.....ll..ll.....',
      '....lllLLlll....',
      '.....lLrrLl.....',
      '......rrrr......',
      '.....rrrrrr.....',
      '.....reRRer.....',
      '.....rrrrrr.....',
      '.....rRrrRr.....',
      '......rrrr......',
      '.....RrrrrR.....',
      '....Rr.rr.rR....',
      '...R...rr...R...',
      '.....RR...R.....',
      '....RR.....RR...',
    ],
    walk_b: [
      '................',
      '......l..l......',
      '.....ll..ll.....',
      '....lllLLlll....',
      '.....lLrrLl.....',
      '......rrrr......',
      '.....rrrrrr.....',
      '.....reRRer.....',
      '.....rrrrrr.....',
      '.....rRrrRr.....',
      '......rrrr......',
      '.....RrrrrR.....',
      '....Rr.rr.rR....',
      '...R...rr...R...',
      '.....R...RR.....',
      '...RR.....RR....',
    ],

    burn_a: [
      '.....f.f.f......',
      '....fFf.fFf.....',
      '....FLl.lLF.....',
      '....lllLLlll....',
      '.....lLrrLl.....',
      '......rrrr......',
      '.....rrrrrr.....',
      '.....reRRer.....',
      '.....rrrrrr.....',
      '.....rRrrRr.....',
      '......rrrr......',
      '.....RrrrrR.....',
      '....Rr.rr.rR....',
      '...R...rr...R...',
      '......R..R......',
      '.....RR..RR.....',
    ],
    burn_b: [
      '......f.f.f.....',
      '.....fFf.fFf....',
      '.....Ll.lL......',
      '....lllLLlll....',
      '.....lLrrLl.....',
      '......rrrr......',
      '.....rrrrrr.....',
      '.....reRRer.....',
      '.....rrrrrr.....',
      '.....rRrrRr.....',
      '......rrrr......',
      '.....RrrrrR.....',
      '....Rr.rr.rR....',
      '...R...rr...R...',
      '......R..R......',
      '.....RR..RR.....',
    ],

    mug: [
      'MMMMMMMMM..',
      'McccccccM..',
      'mmmmmmmmm..',
      'mmmmmmmmmMM',
      'mmmmmmmmm.M',
      'mmmmmmmmmMM',
      '.MmmmmmM...',
      '..MMMMM....',
    ],
  },

  /* Always drawn, always in front, never moves. Everything the creature does
     is relative to it. */
  props: [
    { x: 13, y: 14, front: true, pick: () => 'mug' },
  ],

  states: {
    idle: {
      frames: ['sub_a', 'sub_b'],
      fps: 0.6,
      motion: (v) => ({ x: v.home.x, y: v.home.y + Math.sin(v.t * 0.9) * 0.4 }),
    },

    /* Up far enough to see over the rim. The legs are still in the coffee. */
    work: {
      frames: ['stand'],
      motion: (v) => ({ x: v.home.x, y: 3 + Math.sin(v.t * 2.2) * 0.5 }),
    },

    waiting: {
      frames: ['stand'],
      motion: (v) => ({ x: v.home.x, y: 3 }),
    },

    /* Out of the mug and pacing the desk. */
    grinding: {
      frames: ['walk_a', 'stand', 'walk_b', 'stand'],
      fps: 6,
      motion: (v) => {
        const out = Math.min(1, v.t / 1.1);
        const pace = Math.max(0, v.t - 1.1);
        return { x: 9 - out * 4 - ((1 - Math.cos(pace * 1.6)) / 2) * 5, y: 6 };
      },
    },

    success: {
      frames: ['stand'],
      once: 1300,
      motion: (v) => ({ x: v.home.x, y: 3 - Math.abs(Math.sin(v.p * Math.PI * 2)) * 3 }),
    },

    error: {
      frames: ['burn_a', 'burn_b'],
      fps: 10,
      once: 1900,
      motion: (v) => ({ x: v.home.x + (Math.sin(v.t * 24) > 0 ? 1 : -1), y: 3 }),
    },
  },
};
