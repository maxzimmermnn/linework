"""Watch folder: every photo dropped into a folder gets drawn, the drawing saved next to it.

Polls instead of using filesystem events (no extra dependency, works on network drives).
A file is picked up once its size and mtime are unchanged between two polls, so photos
that are still being copied are left alone. Only photos added after the start count.
"""

from __future__ import annotations

import threading
from pathlib import Path

from . import detect, export

POLL = 2.0  # seconds


def _photos(folder: Path) -> dict[Path, tuple[int, float]]:
    out = {}
    for f in folder.iterdir():
        # skip hidden/partial files and our own drawings ("<photo>_lines_<model>.png")
        if f.suffix.lower() in detect.IMG_EXT and not f.name.startswith(".") and "_lines_" not in f.stem:
            try:
                st = f.stat()
                out[f] = (st.st_size, st.st_mtime)
            except OSError:
                pass
    return out


class Watcher:
    def __init__(self, folder: Path, params: dict, fmt: str, on_event=print):
        export.check(fmt, params)
        self.folder, self.params, self.fmt, self.on_event = folder, dict(params), fmt, on_event
        self.done, self.failed, self.last = 0, 0, ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Watcher":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def _run(self):
        seen = _photos(self.folder)  # already there at start: never processed
        handled = set(seen)
        while not self._stop.wait(POLL):
            try:
                now = _photos(self.folder)
            except OSError as e:  # folder renamed, drive unplugged, ...
                self.last = f"Can't read the folder: {e}"
                continue
            for f, sig in now.items():
                if f in handled or seen.get(f) != sig:
                    continue  # old, or still being written
                handled.add(f)
                self._draw(f)
            seen = now

    def _draw(self, f: Path):
        try:
            rgb = detect.load_image(f)
            edges = detect.detect(rgb, self.params)
            dest, _ = export.save(self.fmt, f, rgb, edges, self.params, f.parent)
            self.done += 1
            self.last = f"{f.name} → {dest.name}"
        except Exception as e:  # one bad file must not stop the watcher
            self.failed += 1
            self.last = f"Could not draw {f.name}: {e}"
        self.on_event(self.last)
