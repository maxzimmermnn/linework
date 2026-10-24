"""Linework in its own window instead of a browser tab (used by Linework.app).

    python -m linework.desktop

Opens a native window right away, starts the local server in the background and
shows the UI when it is ready. Closing the window or Cmd+Q quits everything.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

PAGE = """<html><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
font:15px -apple-system,system-ui,sans-serif;color:#57534e;background:#fafaf9">{}</body></html>"""


def _name_app():
    # Inside Linework.app the running Python still reports its own nameless bundle, so the
    # menu bar would say "Python". pywebview also reads this name for About/Hide/Quit.
    try:
        from Foundation import NSBundle

        NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "Linework"
    except Exception:
        pass


def main():
    _name_app()
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True  # the Download buttons in the UI
    window = webview.create_window("Linework", html=PAGE.format("Starting Linework…"),
                                   width=1440, height=960, min_size=(900, 640))
    window.events.loaded += lambda: print("window loaded", window.get_current_url(), flush=True)

    def serve():
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            import app  # slow: loads torch and gradio

            window.load_url(app.launch(block=False))
        except Exception:
            traceback.print_exc()
            window.load_html(PAGE.format("Linework could not start. Details are in ~/Library/Logs/Linework.log"))

    webview.start(serve, icon=os.environ.get("LINEWORK_ICON"))
    os._exit(0)  # window closed: stop the server threads as well


if __name__ == "__main__":
    main()
