/* The ink slime.
 *
 * Jelly physics inside a sixteen-pixel box: it squashes when it lands,
 * stretches when it leaves the ground, and spreads out flat into a puddle
 * when there is nothing to do but wait for you.
 *
 * It is also the only companion that changes colour. The frames never change
 * for that — one palette entry does — which is why waiting, failing and
 * succeeding can each have their own hue without three sets of art.
 */

export default {
  id: 'slime',
  label: 'Ink slime',
  blurb: 'squashes, stretches, recolours',

  home: { x: 10, y: 8 },
  portrait: { frame: 'blob_a' },

  palette: {
    s: '#3f86cf',   // body
    S: '#2a5f9c',   // where it meets the ground
    h: '#a9d8ff',   // the light coming through the top of it
    e: '#0e2035',   // eyes
    b: '#cfe8ff',   // bubble
  },

  variants: {
    warn: { s: '#d9a441', S: '#9c7223', h: '#ffe6b0' },
    bad: { s: '#d95f4a', S: '#9c3a2a', h: '#ffc4b5' },
    good: { s: '#4bbd7f', S: '#2c8055', h: '#c2f2d8' },
  },

  frames: {
    /* Resting shape. */
    blob_a: [
      '................',
      '................',
      '................',
      '......ssss......',
      '....sshhhsss....',
      '...sshhssssss...',
      '..ssshsssssssss.',
      '..sssssssssssss.',
      '.ssseesssseesss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.SssssssssssssS.',
      '..SSSSSSSSSSSS..',
      '................',
      '................',
    ],

    /* Landed. */
    blob_b: [
      '................',
      '................',
      '................',
      '................',
      '................',
      '.....sssss......',
      '...sshhhsssss...',
      '..sshhsssssssss.',
      '.ssseesssseesss.',
      '.ssssssssssssss.',
      'ssssssssssssssss',
      'ssssssssssssssss',
      'SssssssssssssssS',
      '.SSSSSSSSSSSSSS.',
      '................',
      '................',
    ],

    /* Mid-air. */
    blob_c: [
      '................',
      '.......ss.......',
      '......ssss......',
      '.....sshhs......',
      '....sshhhss.....',
      '....sshsssss....',
      '...ssssssssss...',
      '..ssssssssssss..',
      '..sseesssseess..',
      '..ssssssssssss..',
      '..ssssssssssss..',
      '...ssssssssss...',
      '...SssssssssS...',
      '....SSSSSSSS....',
      '................',
      '................',
    ],

    /* Asleep, and then asleep with a bubble. */
    sleep_a: [
      '................',
      '................',
      '................',
      '......ssss......',
      '....sshhhsss....',
      '...sshhssssss...',
      '..ssshsssssssss.',
      '..sssssssssssss.',
      '.sssSSssssSSsss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.SssssssssssssS.',
      '..SSSSSSSSSSSS..',
      '................',
      '................',
    ],
    sleep_b: [
      '................',
      '................',
      '...........bb...',
      '......ssss.bb...',
      '....sshhhsss....',
      '...sshhssssss...',
      '..ssshsssssssss.',
      '..sssssssssssss.',
      '.sssSSssssSSsss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.SssssssssssssS.',
      '..SSSSSSSSSSSS..',
      '................',
      '................',
    ],
    sleep_c: [
      '.........bb.....',
      '........b..b....',
      '........b..b....',
      '......ssssbb....',
      '....sshhhsss....',
      '...sshhssssss...',
      '..ssshsssssssss.',
      '..sssssssssssss.',
      '.sssSSssssSSsss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.ssssssssssssss.',
      '.SssssssssssssS.',
      '..SSSSSSSSSSSS..',
      '................',
      '................',
    ],

    /* Given up standing. The question mark is drawn in the body colour, so it
       changes with the slime rather than needing its own. */
    puddle_q: [
      '.......sss......',
      '......s...s.....',
      '..........s.....',
      '.........s......',
      '........s.......',
      '................',
      '........s.......',
      '................',
      '......ssss......',
      '...ssssssssss...',
      '.ssssssssssssss.',
      'sssseesssseessss',
      'ssssssssssssssss',
      'SSSSSSSSSSSSSSSS',
      '................',
      '................',
    ],
  },

  states: {
    idle: {
      frames: ['sleep_a', 'sleep_b', 'sleep_c', 'sleep_b'],
      fps: 0.7,
      motion: (v) => ({ x: v.home.x, y: v.home.y + Math.sin(v.t * 1.1) * 0.4 }),
    },

    /* Every keystroke lands on it. */
    typing: {
      frames: ['blob_b', 'blob_c'],
      fps: 15,
      motion: (v) => ({ x: v.home.x + Math.sin(v.t * 40) * 0.5, y: v.home.y }),
    },

    work: {
      frames: ['blob_b', 'blob_c', 'blob_a', 'blob_c'],
      fps: 6,
      motion: (v) => ({ x: v.home.x, y: v.home.y - Math.abs(Math.sin(v.t * 4.7)) * 2.2 }),
    },

    grinding: {
      frames: ['blob_b', 'blob_c', 'blob_a', 'blob_c'],
      fps: 13,
      motion: (v) => ({
        x: v.home.x + Math.sin(v.t * 7.3) * 2,
        y: v.home.y - Math.abs(Math.sin(v.t * 10.2)) * 3,
      }),
    },

    waiting: {
      frames: ['puddle_q'],
      variant: 'warn',
      motion: (v) => ({ x: v.home.x, y: v.home.y + Math.sin(v.t * 1.4) * 0.4 }),
    },

    success: {
      frames: ['blob_c', 'blob_a'],
      variant: 'good',
      fps: 5,
      once: 1300,
      motion: (v) => ({ x: v.home.x, y: v.home.y - Math.abs(Math.sin(v.p * Math.PI * 2)) * 5 }),
    },

    error: {
      frames: ['blob_b'],
      variant: 'bad',
      once: 1300,
      motion: (v) => ({ x: v.home.x + Math.sin(v.t * 33) * 1.2, y: v.home.y }),
    },
  },
};
