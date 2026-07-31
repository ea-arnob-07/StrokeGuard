"""Cross-platform local preview launcher for StrokeGuard."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APP_FILE = APP_DIR / "app.py"
CREDIT_NAME = "Estiuk Arafat Arnob"


def port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def select_port(preferred: int) -> int:
    if port_is_available(preferred):
        return preferred
    for port in range(preferred + 1, preferred + 101):
        if port_is_available(port):
            return port
    raise RuntimeError("No free local preview port was found.")


def wait_until_ready(process: subprocess.Popen[bytes], health_url: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Streamlit exited before preview startup (code {process.returncode})."
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError("The preview server did not become healthy within 90 seconds.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the StrokeGuard preview.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the preview without opening the default browser.",
    )
    args = parser.parse_args()

    port = select_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    preview_url = f"{base_url}/results?demo=1"
    health_url = f"{base_url}/_stcore/health"

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"Starting StrokeGuard by {CREDIT_NAME} on {base_url}")
    process = subprocess.Popen(command, cwd=APP_DIR)
    try:
        wait_until_ready(process, health_url)
        print(f"Preview ready: {preview_url}")
        if not args.no_browser:
            webbrowser.open(preview_url, new=2)
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping preview...")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
