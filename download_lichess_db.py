#!/usr/bin/env python3
"""Download a monthly lichess standard rated games database.

Usage:
    python download_lichess_db.py --month 2026-05

Saves to data/lichess_db_standard_rated_<month>.pgn.zst
"""

import argparse
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
CHUNK_SIZE = 1024 * 1024
FREE_MARGIN = 100 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a monthly lichess standard rated games database"
    )
    parser.add_argument("--month", required=True, help="month in YYYY-MM format, e.g. 2026-05")
    return parser.parse_args()


def validate_month(month):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        sys.exit(f"error: --month must be in YYYY-MM format, got '{month}'")


def head(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return {
            "size": int(response.headers.get("Content-Length", 0)),
            "ranges": response.headers.get("Accept-Ranges", "").lower() == "bytes",
        }


def format_size(n):
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.1f} {unit}"
        size /= 1024


def format_eta(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress(downloaded, total, session_bytes, started_at):
    elapsed = max(time.monotonic() - started_at, 1e-9)
    speed = (downloaded - session_bytes) / elapsed
    eta = (total - downloaded) / speed if speed > 0 else 0
    line = (
        f"{format_size(downloaded)} / {format_size(total)} "
        f"({downloaded / total * 100:5.1f}%)  "
        f"{format_size(speed)}/s  ETA {format_eta(eta)}"
    )
    sys.stdout.write("\r" + line.ljust(90))
    sys.stdout.flush()


def main():
    args = parse_args()
    validate_month(args.month)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, f"lichess_db_standard_rated_{args.month}.pgn.zst")
    url = BASE_URL.format(month=args.month)

    try:
        remote = head(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            sys.exit(f"error: {url} not found - this month is not published on database.lichess.org")
        raise
    except urllib.error.URLError as error:
        sys.exit(f"error: cannot reach database.lichess.org: {error.reason}")

    local_size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if local_size >= remote["size"]:
        print(f"{dest} already complete ({format_size(remote['size'])})")
        return

    remaining = remote["size"] - local_size
    free = shutil.disk_usage(data_dir).free
    if free < remaining + FREE_MARGIN:
        sys.exit(
            f"error: not enough free disk space in {data_dir}: "
            f"need {format_size(remaining)} more, have {format_size(free)} free"
        )

    resume = local_size > 0 and remote["ranges"]
    if local_size > 0 and not resume:
        print("server does not support resume, restarting download from scratch")
        open(dest, "wb").close()
        local_size = 0

    print(f"downloading {url}")
    print(f"saving to {dest}")
    if resume:
        print(f"resuming from {format_size(local_size)}")

    headers = {"Range": f"bytes={local_size}-"} if resume else {}
    request = urllib.request.Request(url, headers=headers)
    started_at = time.monotonic()
    session_bytes = local_size

    try:
        with urllib.request.urlopen(request) as response:
            if resume and response.status != 206:
                print("server ignored resume range, restarting from scratch")
                mode = "wb"
            else:
                mode = "ab"
            with open(dest, mode) as fh:
                downloaded = local_size
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    render_progress(downloaded, remote["size"], session_bytes, started_at)
    except KeyboardInterrupt:
        print()
        print("interrupted - partial file kept, rerun the same command to resume")
        sys.exit(130)
    except urllib.error.HTTPError as error:
        if error.code == 416:
            sys.exit("error: server rejected resume range - delete the partial file and rerun")
        raise

    print()
    print(f"done: {dest} ({format_size(remote['size'])})")


if __name__ == "__main__":
    main()
