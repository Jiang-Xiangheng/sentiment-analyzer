"""Double-clickable launcher for the sentiment analysis web app.

The executable built from this script is a thin wrapper: it starts the project
virtualenv's ``uvicorn`` with the right ``MODEL_PATH``, waits until ``/health``
is ready, then opens the browser. It deliberately does not bundle torch /
transformers, which would make the executable hundreds of megabytes.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010
STARTUP_TIMEOUT = 240.0

# Make status messages visible immediately, even when stdout is redirected.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(line_buffering=True)


def app_root() -> Path:
    """Locate the project root, whether the exe sits in the root or in dist/."""
    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
    else:
        start = Path(__file__).resolve().parent.parent

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return start


def find_python(root: Path) -> str | None:
    for candidate in (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("python") or shutil.which("python3")


def health_ok(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def wait_healthy(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health_ok(host, port):
            return True
        time.sleep(1)
    return False


def pause(message: str = "\nPress Enter to close...") -> None:
    with contextlib.suppress(EOFError):
        input(message)


def _browser_candidates() -> list[tuple[Path, str]]:
    """Return likely browser executables and the flag to open a new window."""
    roots = [
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    relative = [
        ("Microsoft/Edge/Application/Microsoft Edge.exe", "--new-window"),
        ("Google/Chrome/Application/chrome.exe", "--new-window"),
        ("Mozilla Firefox/firefox.exe", "-new-window"),
    ]
    candidates: list[tuple[Path, str]] = []
    for root in roots:
        if not root:
            continue
        for rel, flag in relative:
            candidates.append((Path(root) / rel, flag))

    # Edge is sometimes only available through a versioned launcher directory.
    for root in roots:
        if not root:
            continue
        edge_dir = Path(root) / "Microsoft/Edge/Application"
        if edge_dir.is_dir():
            for versioned in sorted(edge_dir.glob("*/msedge.exe"), reverse=True):
                candidates.append((versioned, "--new-window"))
    return candidates


def open_browser(url: str) -> None:
    """Open the URL with the system default browser and bring it to the front."""
    if sys.platform == "win32":
        for executable, flag in _browser_candidates():
            if not executable.is_file():
                continue
            try:
                subprocess.Popen(
                    [str(executable), flag, url],
                    close_fds=True,
                )
                return
            except OSError:
                continue
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", url],
                close_fds=True,
            )
            return
        except OSError:
            pass
    try:
        if hasattr(os, "startfile"):
            os.startfile(url)  # type: ignore[attr-defined]
            return
    except OSError:
        pass
    with contextlib.suppress(Exception):
        webbrowser.open(url, new=1, autoraise=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the sentiment analysis web app.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    root = app_root()

    print("=" * 62)
    print("  Sentiment Analyzer")
    print("=" * 62)
    print(f"  project : {root}")

    if health_ok(args.host, args.port):
        print("  server  : already running")
        print(f"  open    : {url}")
        if not args.no_browser:
            open_browser(url)
        print(f"  If the browser did not open, visit: {url}")
        return 0

    python = find_python(root)
    if python is None:
        print("\n[ERROR] Python environment not found.")
        print("        Run:  uv sync --all-groups")
        pause()
        return 1

    model = root / "artifacts" / "checkpoints" / "best.ckpt"
    if not model.is_file():
        print(f"\n[WARN] model not found: {model}")
        print("       The UI will open, but /classify will return 503.")

    env = os.environ.copy()
    env["MODEL_PATH"] = str(model)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep the console readable for non-technical users.
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"  python  : {python}")
    print(f"  model   : {model}")
    print(f"  address : {url}")
    print("-" * 62)

    process = subprocess.Popen(
        [
            python,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=str(root),
        env=env,
    )

    try:
        print("  loading the model, this can take ~30 seconds ...")
        if not wait_healthy(args.host, args.port, STARTUP_TIMEOUT):
            print("\n[ERROR] the server did not become healthy in time.")
            print("        Check the log above for details.")
            pause()
            return 1

        print(f"\n  ready: {url}")
        if not args.no_browser:
            open_browser(url)
        print(f"  If the browser did not open, visit: {url}")
        print("  Press Ctrl+C to stop the server.\n")
        process.wait()
    except KeyboardInterrupt:
        print("\n  stopping ...")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
