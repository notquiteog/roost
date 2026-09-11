/* The mech-bit.
 *
 * A CRT on chicken legs. Its face is a screen, which makes it the one
 * companion that can say something specific: a scanline while the model is
 * thinking, its drive light while files are being read, and a progress bar
 * filling across its face while a command runs.
 *
 * The screen is a separate sprite drawn over the case rather than a set of
 * whole-body frames, so a new expression is five rows of eight characters
 * instead of another sixteen-by-sixteen.
 */

export default {
  id: 'mechbit',
  label: 'Mech-bit',
  blurb: 'a CRT on chicken legs',

  home: { x: 10, y: 8 },
  portrait: { frame: 'body', props: [['face_eyes', 4, 3]] },

  palette: {
    c: '#c2bbab',   // case
    C: '#8f8879',   // case, drive bay
    d: '#5a554d',   // case, underside
    h: '#e9e4d6',   // bezel
    k: '#101c18',   // screen, unlit
    l: '#d3a24c',   // legs
    p: '#6ee89a',   // phosphor
    P: '#2f6b46',   // phosphor, burnt in
  },

  variants: {
    bad: { p: '#ff7a6a', P: '#7a2f28' },
  },

  frames: {
    body: [
      '................',
      '..cccccccccccc..',
      '..chhhhhhhhhhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chhhhhhhhhhc..',
      '..cCCCCCCCCCCc..',
      '..dddddddddddd..',
      '....l......l....',
      '....l......l....',
      '...lll....lll...',
      '................',
      '................',
    ],
    walk_a: [
      '................',
      '..cccccccccccc..',
      '..chhhhhhhhhhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chhhhhhhhhhc..',
      '..cCCCCCCCCCCc..',
      '..dddddddddddd..',
      '....l......l....',
      '...l........l...',
      '..lll......lll..',
      '................',
      '................',
    ],
    walk_b: [
      '................',
      '..cccccccccccc..',
      '..chhhhhhhhhhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chkkkkkkkkhc..',
      '..chhhhhhhhhhc..',
      '..cCCCCCCCCCCc..',
      '..dddddddddddd..',
      '....l......l....',
      '.....l....l.....',
      '....lll..lll....',
      '................',
      '................',
    ],

    /* Everything below is drawn on the screen: eight by five, transparent
       where the tube shows through. */
    face_eyes: [
      '........',
      '.pp..pp.',
      '........',
      '.p....p.',
      '..pppp..',
    ],

    face_scan0: ['pppppppp', '........', '........', '........', '........'],
    face_scan1: ['........', 'pppppppp', '........', '........', '........'],
    face_scan2: ['........', '........', 'pppppppp', '........', '........'],
    face_scan3: ['........', '........', '........', 'pppppppp', '........'],
    face_scan4: ['........', '........', '........', '........', 'pppppppp'],

    face_bar0: ['........', '........', 'PPPPPPPP', '........', '........'],
    face_bar1: ['........', '........', 'ppPPPPPP', '........', '........'],
    face_bar2: ['........', '........', 'ppppPPPP', '........', '........'],
    face_bar3: ['........', '........', 'ppppppPP', '........', '........'],
    face_bar4: ['........', '........', 'pppppppp', '........', '........'],

    face_saver0: ['p.......', '........', '........', '........', '........'],
    face_saver1: ['........', '..p.....', '........', '........', '........'],
    face_saver2: ['........', '........', '....p...', '........', '........'],
    face_saver3: ['........', '........', '........', '......p.', '........'],
    face_saver4: ['........', '........', '........', '........', '.......p'],

    face_q: [
      '..ppp...',
      '.p...p..',
      '....pp..',
      '...p....',
      '...p....',
    ],
    face_check: [
      '........',
      '......pp',
      '.....pp.',
      'p...pp..',
      '.pppp...',
    ],
    face_bang: [
      '...pp...',
      '...pp...',
      '...pp...',
      '........',
      '...pp...',
    ],

    drive_on: ['pppppp'],
  },

  props: [
    /* Attached, so the face stays on the face when the whole machine shakes. */
    {
      x: 4,
      y: 3,
      attach: true,
      front: true,
      pick: (v) => {
        switch (v.state) {
          case 'idle':
          case 'typing': {
            // Five frames, played out and back, which is a bouncing pixel.
            const n = Math.floor(v.t * 3) % 8;
            return `face_saver${n < 5 ? n : 8 - n}`;
          }
          case 'waiting':
          case 'listening':
            return 'face_q';
          case 'success':
            return 'face_check';
          case 'error':
            return 'face_bang';
          case 'reading':
            return 'face_eyes';
          case 'running':
          case 'grinding':
            return `face_bar${Math.floor(v.t * 2) % 5}`;
          default:
            return `face_scan${Math.floor(v.t * 10) % 5}`;
        }
      },
    },
    /* The drive light, which only means one thing. */
    {
      x: 5,
      y: 9,
      attach: true,
      front: true,
      pick: (v) => (v.state === 'reading' && Math.floor(v.t * 7) % 2 ? 'drive_on' : null),
    },
  ],

  states: {
    idle: { frames: ['body'] },
    work: { frames: ['body'] },
    reading: { frames: ['body'] },
    waiting: { frames: ['body'] },

    running: {
      frames: ['walk_a', 'walk_b'],
      fps: 5,
      motion: (v) => ({ x: v.home.x, y: v.home.y - (Math.floor(v.t * 5) % 2) }),
    },

    grinding: {
      frames: ['walk_a', 'walk_b'],
      fps: 9,
      motion: (v) => ({ x: v.home.x + Math.sin(v.t * 1.6) * 3, y: v.home.y - (Math.floor(v.t * 9) % 2) }),
    },

    success: { frames: ['body'], once: 1400 },

    error: {
      frames: ['body'],
      variant: 'bad',
      once: 1400,
      motion: (v) => ({ x: v.home.x + (Math.sin(v.t * 30) > 0 ? 1 : -1), y: v.home.y }),
    },
  },
};
