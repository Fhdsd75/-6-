#!/usr/bin/env python3
"""Login to lk.rulionline.ru and download student video tutorials."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests

CDN_BASE = "https://cdn.rulionline.ru/videos/"
API_VIDEOS = "/api/students/videos"
DEFAULT_OUT = "downloaded_videos"
DEFAULT_QUALITY = "full"  # full | original | small
MAX_WORKERS = 2
TIMEOUT = (20, 180)
RETRY_SLEEP_SEC = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://lk.rulionline.ru/",
}

QUALITY_FIELD = {
    "full": "linkFull",
    "original": "linkOriginal",
    "small": "linkSmall",
}

print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def load_local_credentials() -> None:
    candidates = (
        Path(__file__).resolve().parent / ".local_credentials.env",
        Path.cwd() / ".local_credentials.env",
    )
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text or "video")[:max_len]


@dataclass
class VideoItem:
    section: str
    name: str
    video_id: str
    filename_remote: str
    url: str
    fallback_urls: tuple[str, ...] = ()

    @property
    def local_name(self) -> str:
        return f"{safe_name(self.section)} — {safe_name(self.name)}.mp4"

    @property
    def candidate_urls(self) -> list[str]:
        out: list[str] = []
        for url in (self.url, *self.fallback_urls):
            if url and url not in out:
                out.append(url)
        return out


def login_and_fetch_videos(base_url: str, username: str, password: str) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright required: pip install playwright && playwright install chromium"
        ) from exc

    log("→ Логинюсь через браузер…")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                locale="ru-RU",
                viewport={"width": 1365, "height": 900},
            )
            page = context.new_page()
            page.goto(urljoin(base_url, "/videotutorials/"), wait_until="domcontentloaded", timeout=60000)
            page.fill('input[name="username"]', username)
            page.fill('input[name="password"]', password)
            with page.expect_response(
                lambda r: "/api/" in r.url and "login" in r.url.lower() and r.request.method == "POST",
                timeout=45000,
            ) as resp_info:
                page.click('button[type="submit"]')
            resp = resp_info.value
            if resp.status >= 400:
                raise SystemExit(f"✗ Логин отклонён: HTTP {resp.status}")
            page.wait_for_timeout(1500)
            log(f"✓ Вход OK: {page.url} | {page.title()}")

            data = page.evaluate(
                f"""async () => {{
                    const r = await fetch('{API_VIDEOS}', {{credentials:'include'}});
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return await r.json();
                }}"""
            )
        finally:
            browser.close()

    if not isinstance(data, list):
        raise SystemExit(f"✗ Неожиданный ответ API: {type(data)}")
    log(f"✓ Список видео получен: разделов={len(data)}")
    return data


def build_items(groups: list[dict], quality: str) -> list[VideoItem]:
    preferred = QUALITY_FIELD[quality]
    # Prefer requested quality, then smaller/faster fallbacks, then original.
    order = [preferred, "linkFull", "linkSmall", "linkOriginal"]
    dedup_order: list[str] = []
    for key in order:
        if key not in dedup_order:
            dedup_order.append(key)

    items: list[VideoItem] = []
    for group in groups:
        section = group.get("name") or "section"
        for video in group.get("videos") or []:
            remotes = [video.get(k) for k in dedup_order if video.get(k)]
            # unique preserve order
            uniq: list[str] = []
            for remote in remotes:
                if remote not in uniq:
                    uniq.append(remote)
            if not uniq:
                continue
            urls = [urljoin(CDN_BASE, remote) for remote in uniq]
            items.append(
                VideoItem(
                    section=section,
                    name=video.get("name") or video.get("_id") or "video",
                    video_id=str(video.get("_id") or ""),
                    filename_remote=uniq[0],
                    url=urls[0],
                    fallback_urls=tuple(urls[1:]),
                )
            )
    return items


def unique_path(out_dir: Path, item: VideoItem, used: set[str]) -> Path:
    base = item.local_name
    if base not in used:
        used.add(base)
        return out_dir / base
    stem = Path(base).stem
    n = 2
    while True:
        candidate = f"{stem} ({n}).mp4"
        if candidate not in used:
            used.add(candidate)
            return out_dir / candidate
        n += 1


def download_one(
    session: requests.Session,
    item: VideoItem,
    path: Path,
    retries: int = 8,
) -> tuple[str, str]:
    """Download via curl resume — more stable than requests on this CDN."""
    import subprocess
    import time

    del session  # unused; kept for call-site compatibility
    part = path.with_suffix(path.suffix + ".part")
    if path.exists() and path.stat().st_size > 0 and not part.exists():
        return item.local_name, "skip"

    last_err = "unknown"
    for url in item.candidate_urls:
        # Probe availability quickly
        try:
            probe = requests.head(url, headers=HEADERS, timeout=20, allow_redirects=True)
            if probe.status_code == 404:
                last_err = f"HTTP 404 for {url}"
                continue
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)

        for attempt in range(1, retries + 1):
            cmd = [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "-C",
                "-",
                "-A",
                HEADERS["User-Agent"],
                "-H",
                f"Referer: {HEADERS['Referer']}",
                "-o",
                str(part),
                url,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    err = (proc.stderr or proc.stdout or "").strip().splitlines()
                    last_err = err[-1] if err else f"curl exit {proc.returncode}"
                    # 404 from curl
                    if "404" in last_err or proc.returncode == 22:
                        break
                    log(f"• retry {attempt}/{retries} for {item.local_name}: {last_err}")
                    time.sleep(RETRY_SLEEP_SEC)
                    continue

                if not part.exists() or part.stat().st_size <= 0:
                    last_err = "empty file"
                    time.sleep(RETRY_SLEEP_SEC)
                    continue

                part.replace(path)
                mb = path.stat().st_size / (1024 * 1024)
                suffix = "" if url == item.url else " (fallback)"
                return item.local_name, f"ok {mb:.1f} MB{suffix}"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                log(f"• retry {attempt}/{retries} for {item.local_name}: {last_err}")
                time.sleep(RETRY_SLEEP_SEC)
                continue

    return item.local_name, f"fail {last_err}"


def parse_args() -> argparse.Namespace:
    load_local_credentials()
    p = argparse.ArgumentParser(description="Скачать видеоуроки с lk.rulionline.ru")
    p.add_argument(
        "--base-url",
        default="https://lk.rulionline.ru",
        help="Базовый URL кабинета",
    )
    p.add_argument(
        "--username",
        default=os.getenv("FIND_VIDEO_API_USER") or os.getenv("LK_USERNAME"),
    )
    p.add_argument(
        "--password",
        default=os.getenv("FIND_VIDEO_API_PASSWORD") or os.getenv("LK_PASSWORD"),
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="Папка для сохранения")
    p.add_argument(
        "--quality",
        choices=sorted(QUALITY_FIELD),
        default=DEFAULT_QUALITY,
        help="full≈4GB всего, small≈1.3GB, original≈33GB",
    )
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Только показать список, не скачивать",
    )
    p.add_argument(
        "--manifest",
        default="",
        help="Куда сохранить JSON-манифест (по умолчанию OUT/manifest.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.username or not args.password:
        raise SystemExit("Нужны username/password или scripts/.local_credentials.env")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = login_and_fetch_videos(args.base_url, args.username, args.password)
    items = build_items(groups, args.quality)
    log(f"→ К скачиванию: {len(items)} файлов, quality={args.quality}, workers={args.workers}")

    used: set[str] = set()
    planned: list[tuple[VideoItem, Path]] = []
    for item in items:
        planned.append((item, unique_path(out_dir, item, used)))

    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    manifest = [
        {
            "section": i.section,
            "name": i.name,
            "id": i.video_id,
            "remote": i.filename_remote,
            "url": i.url,
            "local": str(path.name),
        }
        for i, path in planned
    ]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"✓ Манифест: {manifest_path}")

    if args.list_only:
        for row in manifest:
            print(f"- {row['local']} <= {row['url']}")
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, session, item, path): (item, path)
            for item, path in planned
        }
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            name, status = fut.result()
            if status == "skip":
                skip += 1
                mark = "•"
            elif status.startswith("ok") or status == "done":
                ok += 1
                mark = "✓"
            else:
                fail += 1
                mark = "✗"
            log(f"{mark} [{done}/{total}] {name}: {status}")

    log("=" * 60)
    log(f"Готово: скачано={ok}, пропущено={skip}, ошибки={fail}, папка={out_dir.resolve()}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
