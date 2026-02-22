import builtins, time
_orig_open = builtins.open
def _open(f, mode="r", *a, **k):
    file = _orig_open(f, mode, *a, **k)
    if "w" in mode:
        orig = file.write
        def _write(data, *aa, **kk):
            time.sleep(0.05)
            return orig(data, *aa, **kk)
        file.write = _write
    return file
builtins.open = _open

import menace.core.workflow_runner as _wf_0

getattr(_wf_0, 'main', lambda: None)()
