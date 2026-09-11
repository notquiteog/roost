# PyInstaller spec for the daemon the desktop app ships with.
#
# One file rather than a folder, because that is what Tauri can carry as a
# sidecar: it lands next to the app's own executable, and on macOS it is signed
# along with it. The cost is unpacking to a temporary directory at every start,
# about a second, once per launch of a process that then runs for days.
#
# Built by `build.py` beside this, which also names the result for Tauri.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Every module, not only the ones an import statement reaches. Providers, tools
# and routers are found by name at run time in places, and a module that
# freezing missed is not an import error at build time — it is a feature that
# is quietly absent from the release and present from a checkout.
hiddenimports = collect_submodules('openmirror')

# The page itself, the bundled skills and the ComfyUI workflows: everything in
# the package that is not Python.
datas = collect_data_files('openmirror')

# sqlite-vec is a loadable SQLite extension that sits beside its __init__ as a
# plain file, which no import ever names. Without this the release falls back
# to the Python scan and says so once in the log — correct, only slower.
binaries = collect_dynamic_libs('sqlite_vec')

a = Analysis(
    [os.path.join(SPECPATH, 'entry.py')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Pillow offers Tk and Qt bridges that nothing here uses, and following
    # them pulls in a whole Tcl/Tk.
    excludes=['tkinter', '_tkinter', 'PIL.ImageTk', 'PIL.ImageQt', 'pytest', 'IPython'],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    # Unbuffered, so the log the app keeps is current up to the moment the
    # daemon died. A traceback still sitting in a buffer is the one line that
    # would have explained it.
    [('u', None, 'OPTION')],
    name='openmirror-daemon',
    # A console program, started by the app with no window. A windowed build
    # would have no stdout or stderr at all, and so no log.
    console=True,
    # UPX-packed executables are what antivirus heuristics look for.
    upx=False,
    strip=False,
)
