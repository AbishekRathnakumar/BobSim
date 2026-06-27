from __future__ import annotations

from contextlib import closing
import importlib.util
import os
import socket
import sys
import threading
import time
import webbrowser

from _5_App import app as bobsim_app


APP_TITLE = "BobDyn"
HOST = "127.0.0.1"


def _available_port(host: str) -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _run_server(host: str, port: int) -> None:
    bobsim_app.run(host, port)


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _has_qt_webengine() -> bool:
    return _has_module("PyQt6.QtWebEngineCore") and _has_module("PyQt6.QtWebEngineWidgets")


def _can_start_embedded_window() -> bool:
    if sys.platform.startswith("linux"):
        return _has_module("gi") or _has_qt_webengine()
    return True


def _preferred_webview_gui() -> str | None:
    if sys.platform.startswith("linux") and _has_qt_webengine():
        return "qt"
    return None


def _open_browser_and_wait(url: str, server_thread: threading.Thread) -> None:
    print(f"Opening BobDyn in your browser: {url}", flush=True)
    webbrowser.open(url)
    try:
        while server_thread.is_alive():
            time.sleep(3600)
    except KeyboardInterrupt:
        return


def main() -> None:
    port = _available_port(HOST)
    url = f"http://{HOST}:{port}"
    server_thread = threading.Thread(target=_run_server, args=(HOST, port), daemon=True)
    server_thread.start()
    time.sleep(0.35)

    os.environ.setdefault("QT_API", "pyqt6")

    if not _can_start_embedded_window():
        _open_browser_and_wait(url, server_thread)
        return

    try:
        import webview
    except Exception as exc:
        print(f"Embedded BobDyn window unavailable: {exc}", flush=True)
        _open_browser_and_wait(url, server_thread)
        return

    try:
        webview.create_window(APP_TITLE, url, width=1440, height=960, min_size=(1100, 700))
        webview.start(gui=_preferred_webview_gui())
    except Exception as exc:
        print(f"Embedded BobDyn window failed to start: {exc}", flush=True)
        _open_browser_and_wait(url, server_thread)


if __name__ == "__main__":
    main()
