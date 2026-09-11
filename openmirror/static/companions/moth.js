/* The poly-moth.
 *
 * The oldest joke in the trade, drawn at sixteen pixels: a literal bug. It
 * spends a working turn flying at a lamp, hovers when the agent is waiting on
 * you, and when something throws it flies straight into the side of the
 * window and slides down it.
 */

export default {
  id: 'moth',
  label: 'Poly-moth',
  blurb: 'a literal bug, drawn to the light',

  home: { x: 3, y: 6 },
  portrait: { frame: 'rest_open' },

  palette: {
    a: '#2f281f',   // antennae
    b: '#4b3f31',   // body
    w: '#e0cda6',   // wing, lit face
    W: '#9c8763',   // wing, shaded edge
    e: '#f7f2e6',   // eye
    x: '#1b1512',   // eye, after the wall
    o: '#ffe6a0',   // bulb, lit
    O: '#5f5a4c',   // bulb, cold
    m: '#9a9488',   // fitting
    M: '#5d594f',   // fitting, shaded
  },

  frames: {
    /* Wings held high, the top of a beat. */
    fly_up: [
      '...a........a...',
      '....a......a....',
      '.....a....a.....',
      '.....bbbbbb.....',
      '.....ebbbbe.....',
      '.WWW..bbbb..WWW.',
      'WwwwW.bbbb.WwwwW',
      'WwwwwWbbbbWwwwwW',
      'WwwwwWbbbbWwwwwW',
      '.WwwwWbbbbWwwwW.',
      '..WwwWbbbbWwwW..',
      '...WWWbbbbWWW...',
      '......bbbb......',
      '.......bb.......',
      '.......bb.......',
      '................',
    ],
    fly_mid: [
      '...a........a...',
      '....a......a....',
      '.....a....a.....',
      '.....bbbbbb.....',
      '.....ebbbbe.....',
      '..WW..bbbb..WW..',
      '.WwwW.bbbb.WwwW.',
      '.WwwwWbbbbWwwwW.',
      'WwwwwWbbbbWwwwwW',
      '.WwwwWbbbbWwwwW.',
      '..WwwWbbbbWwwW..',
      '...WWWbbbbWWW...',
      '......bbbb......',
      '.......bb.......',
      '.......bb.......',
      '................',
    ],
    fly_down: [
      '...a........a...',
      '....a......a....',
      '.....a....a.....',
      '.....bbbbbb.....',
      '.....ebbbbe.....',
      '......bbbb......',
      '...WW.bbbb.WW...',
      '..WwwWbbbbWwwW..',
      '.WwwwWbbbbWwwwW.',
      '.WwwwWbbbbWwwwW.',
      '..WwwWbbbbWwwW..',
      '...WWWbbbbWWW...',
      '....W.bbbb.W....',
      '.......bb.......',
      '.......bb.......',
      '................',
    ],

    /* At rest the wings fold back over the body, and the whole animation is
       them opening a little and closing again. */
    rest_closed: [
      '................',
      '...a........a...',
      '....a......a....',
      '.....a....a.....',
      '.....bbbbbb.....',
      '.....ebbbbe.....',
      '.....WbbbbW.....',
      '....WwbbbbwW....',
      '....WwbbbbwW....',
      '....WwwbbwwW....',
      '.....WwbbwW.....',
      '.....WwbbwW.....',
      '......Wbbw......',
      '.......bb.......',
      '................',
      '................',
    ],
    rest_open: [
      '................',
      '...a........a...',
      '....a......a....',
      '.....a....a.....',
      '.....bbbbbb.....',
      '.....ebbbbe.....',
      '....WWbbbbWW....',
      '...WwwbbbbwwW...',
      '..WwwwbbbbwwwW..',
      '...WwwwbbwwwW...',
      '....WwwbbwwW....',
      '.....WwbbwW.....',
      '......Wbbw......',
      '.......bb.......',
      '................',
      '................',
    ],

    /* Having met the window. */
    dazed: [
      '................',
      '................',
      '...a........a...',
      '....a......a....',
      '.....bbbbbb.....',
      '.....xbbbbx.....',
      '...WW.bbbb.WW...',
      '..WwwWbbbbWwwW..',
      '.WwwwWbbbbWwwwW.',
      '..WwwWbbbbWwwW..',
      '...WWWbbbbWWW...',
      '......bbbb......',
      '.......bb.......',
      '.......bb.......',
      '................',
      '................',
    ],

    bulb_on: [
      '...ooo...',
      '..ooooo..',
      '.ooooooo.',
      '.ooooooo.',
      '.ooooooo.',
      '..ooooo..',
      '...ooo...',
      '..mmmmm..',
      '..MMMMM..',
      '..mmmmm..',
      '...MMM...',
    ],
    bulb_off: [
      '...OOO...',
      '..OOOOO..',
      '.OOOOOOO.',
      '.OOOOOOO.',
      '.OOOOOOO.',
      '..OOOOO..',
      '...OOO...',
      '..mmmmm..',
      '..MMMMM..',
      '..mmmmm..',
      '...MMM...',
    ],
  },

  /* The lamp is the fixed point everything else orbits, so it is drawn behind
     and does not move. It is lit exactly when there is something to look at. */
  props: [
    {
      x: 14,
      y: 2,
      front: false,
      pick: (v) => (v.state === 'idle' || v.state === 'typing' ? 'bulb_off' : 'bulb_on'),
    },
  ],

  states: {
    idle: {
      frames: ['rest_closed', 'rest_open'],
      fps: 0.9,
      motion: (v) => ({ x: v.home.x, y: v.home.y + Math.sin(v.t * 1.3) * 0.6 }),
    },

    /* Around the lamp. Two frequencies rather than a circle, because a moth
       does not fly in a circle — it overshoots the light and comes back. */
    work: {
      frames: ['fly_up', 'fly_mid', 'fly_down', 'fly_mid'],
      fps: 14,
      motion: (v) => {
        const a = v.t * 3.2;
        return { x: 18 + Math.cos(a) * 10 - 8, y: 12 + Math.sin(a * 1.7) * 3 - 8 };
      },
    },

    /* Twenty-five seconds in and it is battering the glass. */
    grinding: {
      frames: ['fly_up', 'fly_mid', 'fly_down', 'fly_mid'],
      fps: 22,
      motion: (v) => {
        const a = v.t * 6.5;
        return {
          x: 18 + Math.cos(a) * 10.5 - 8 + Math.sin(v.t * 31) * 0.7,
          y: 12 + Math.sin(a * 2.3) * 3.5 - 8,
        };
      },
    },

    waiting: {
      frames: ['fly_up', 'fly_mid', 'fly_down', 'fly_mid'],
      fps: 7,
      motion: (v) => ({ x: 10 + Math.sin(v.t * 1.2) * 3, y: 3 + Math.sin(v.t * 2.1) * 0.8 }),
    },

    success: {
      frames: ['fly_up', 'fly_mid', 'fly_down', 'fly_mid'],
      fps: 18,
      once: 1300,
      motion: (v) => ({ x: 3 + v.p * 6, y: 6 - v.p * 5 + Math.sin(v.t * 9) * 0.6 }),
    },

    /* Straight into the side of the window, then down it. */
    error: {
      once: 1600,
      frames: ['fly_up', 'fly_down', 'dazed'],
      pick: (v) => {
        if (v.p >= 0.34) return 'dazed';
        return Math.floor(v.t * 20) % 2 ? 'fly_up' : 'fly_down';
      },
      motion: (v) => {
        const hit = 0.34;
        if (v.p < hit) {
          const k = v.p / hit;
          return { x: 2 + k * 18, y: 5 - k * 2 };
        }
        const k = (v.p - hit) / (1 - hit);
        return { x: 20 - k * 1.5, y: 3 + k * 5 };
      },
    },
  },
};
