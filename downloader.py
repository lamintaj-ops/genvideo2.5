# downloader.py

import requests
from pathlib import Path

def download_file(url, out_path: Path, timeout=60):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    expected = resp.headers.get("Content-Length")
    expected = int(expected) if expected else None

    total = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    if expected and total < expected:
        raise RuntimeError(f"Downloaded incomplete file: got {total}, expected {expected}")

    return total
