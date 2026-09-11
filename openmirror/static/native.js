/* The desktop app, when this page is inside it.

   The page is the same one a browser tab gets. The app adds only what a tab
   cannot do, and the part of that which needs the page is this: a
   notification when the agent stops to wait for you while you are looking at
   something else. The app lives in the tray, and a question asked of a hidden
   window is a question nobody sees.

   Detected by what the app provides, not by user agent — `__TAURI__` exists
   only where the app put it, and the notification API only where the app has
   granted this page the use of it. */

import { caption, companion } from './companions/index.js';

const shell = window.__TAURI__;

if (shell && shell.notification) {
  let was = companion.state;
  companion.watch((state) => {
    // The moment it starts waiting, once — not every time a state that is
    // already "waiting" is announced again.
    if (state === 'waiting' && was !== 'waiting' && !document.hasFocus()) notify(shell.notification);
    was = state;
  });
}

async function notify(api) {
  try {
    let granted = await api.isPermissionGranted();
    if (!granted) granted = (await api.requestPermission()) === 'granted';
    if (!granted) return;
    const words = caption('waiting');
    api.sendNotification({ title: 'openmirror', body: words[0].toUpperCase() + words.slice(1) });
  } catch (err) {
    // Worse than a notification, better than breaking the page over one.
    console.warn('openmirror: could not notify', err);
  }
}
