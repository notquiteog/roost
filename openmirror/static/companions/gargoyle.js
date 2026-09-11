/* The overworked gargoyle.
 *
 * Made of stone, so it moves almost never — which is the point. A creature
 * that reacts to everything stops carrying information; one that has been a
 * statue for four minutes and then opens an eye has your full attention.
 *
 * Its whole idle animation is a vine growing over its shoulder, which is also
 * the only clock in the interface that measures how long you have been
 * thinking about what to type.
 */

export default {
  id: 'gargoyle',
  label: 'Gargoyle',
  blurb: 'stone, until it is not',

  home: { x: 10, y: 6 },
  portrait: { frame: 'still' },

  palette: {
    s: '#8f8b83',   // stone, lit
    S: '#6d6961',   // stone, turned away
    d: '#4a4741',   // stone, in shadow
    e: '#ffcf6b',   // whatever is behind the eyes
    v: '#4f8f47',   // vine
    l: '#78bd63',   // leaf
    c: '#b3aea4',   // dust off the pedestal
  },

  variants: {
    /* One colour changes and the same frame means the opposite thing. */
    alarm: { e: '#ff6a54' },
  },

  frames: {
    still: [
      '................',
      '...d........d...',
      '...dd......dd...',
      '....ssssssss....',
      '...ssssssssss...',
      '...ssSSSSSSss...',
      '...ssssssssss...',
      '....ssSSSSss....',
      '.....ssssss.....',
      '..dddssssssddd..',
      '.dSSssssssssSSd.',
      '.dSSssssssssSSd.',
      '..dSSssssssSSd..',
      '..dddddddddddd..',
      '..SSSSSSSSSSSS..',
      '..dddddddddddd..',
    ],

    /* One eye. The cheapest possible acknowledgement. */
    eye_one: [
      '................',
      '...d........d...',
      '...dd......dd...',
      '....ssssssss....',
      '...ssssssssss...',
      '...sseSSSSSss...',
      '...ssssssssss...',
      '....ssSSSSss....',
      '.....ssssss.....',
      '..dddssssssddd..',
      '.dSSssssssssSSd.',
      '.dSSssssssssSSd.',
      '..dSSssssssSSd..',
      '..dddddddddddd..',
      '..SSSSSSSSSSSS..',
      '..dddddddddddd..',
    ],

    eye_open: [
      '................',
      '...d........d...',
      '...dd......dd...',
      '....ssssssss....',
      '...ssssssssss...',
      '...sseSSSSess...',
      '...ssssssssss...',
      '....ssSSSSss....',
      '.....ssssss.....',
      '..dddssssssddd..',
      '.dSSssssssssSSd.',
      '.dSSssssssssSSd.',
      '..dSSssssssSSd..',
      '..dddddddddddd..',
      '..SSSSSSSSSSSS..',
      '..dddddddddddd..',
    ],

    /* Both eyes, and an arm it has never used before. */
    cheer: [
      '................',
      '...d........d...',
      '...dd......dd...',
      '....ssssssss....',
      '...ssssssssss...',
      '...sseSSSSess...',
      '...ssssssssss.s.',
      '....ssSSSSss.ss.',
      '.....ssssss..ss.',
      '..dddssssssddd..',
      '.dSSssssssssSSd.',
      '.dSSssssssssSSd.',
      '..dSSssssssSSd..',
      '..dddddddddddd..',
      '..SSSSSSSSSSSS..',
      '..dddddddddddd..',
    ],

    vine1: [
      '......',
      '......',
      '......',
      '......',
      '......',
      '...v..',
      '..vv..',
      '..v...',
    ],
    vine2: [
      '......',
      '......',
      '....l.',
      '...vl.',
      '...v..',
      '..vv..',
      '..v...',
      '..v...',
    ],
    vine3: [
      '..l...',
      '.lvl..',
      '..vl..',
      '..v...',
      '..vl..',
      '..vv..',
      '..v...',
      '..v...',
    ],

    dust1: [
      '....',
      '..c.',
      '.c..',
    ],
    dust2: [
      '..c.',
      '.c..',
      'c...',
    ],
  },

  props: [
    /* The vine only grows while nothing is happening, and it is gone the
       moment you press enter. Nobody is told the rule; it is obvious the
       second time you see it. */
    {
      x: 0,
      y: 5,
      attach: true,
      front: true,
      pick: (v) => {
        if (v.state !== 'idle') return null;
        if (v.t < 8) return null;
        if (v.t < 18) return 'vine1';
        if (v.t < 32) return 'vine2';
        return 'vine3';
      },
    },
    /* Chips off its own pedestal while a long build runs. */
    {
      x: 10,
      y: 12,
      attach: true,
      front: true,
      pick: (v) => (v.state === 'grinding' ? (Math.floor(v.t * 6) % 2 ? 'dust1' : 'dust2') : null),
    },
  ],

  states: {
    idle: { frames: ['still'] },

    work: { frames: ['still', 'eye_one'], fps: 0.7 },

    waiting: { frames: ['eye_open'] },

    /* Stone does not pace. It vibrates. */
    grinding: {
      frames: ['still'],
      motion: (v) => ({ x: v.home.x + (Math.sin(v.t * 27) > 0 ? 1 : 0), y: v.home.y }),
    },

    success: { frames: ['cheer'], once: 1700 },

    error: {
      frames: ['eye_open'],
      variant: 'alarm',
      once: 1400,
      motion: (v) => ({ x: v.home.x + (Math.sin(v.t * 34) > 0 ? 1 : -1), y: v.home.y }),
    },
  },
};
