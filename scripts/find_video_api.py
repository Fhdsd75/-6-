#!/usr/bin/env python3
"""Find API endpoints related to video/search in a page and its JS assets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 20
MAX_JS_FILES = 40
MAX_JS_BYTES = 2_000_000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Words that often appear near video-search APIs
VIDEO_HINTS = (
    "video",
    "videos",
    "media",
    "clip",
    "stream",
    "playback",
    "player",
    "mp4",
    "hls",
    "m3u8",
    "vod",
    "watch",
    "thumbnail",
)

SEARCH_HINTS = (
    "search",
    "query",
    "find",
    "suggest",
    "autocomplete",
    "filter",
    "results",
    "catalog",
    "list",
    "feed",
    "browse",
)

API_HINTS = (
    "api",
    "graphql",
    "gql",
    "rest",
    "endpoint",
    "ajax",
    "rpc",
    "v1",
    "v2",
    "v3",
)

# Absolute / relative URL-looking API paths inside JS/HTML
URL_PATTERN = re.compile(
    r"""(?P<quote>['"`])
        (?P<url>
            (?:https?:)?//[^\s'"`<>\\]{3,300}
            |
            /[A-Za-z0-9][A-Za-z0-9_./\-?=&#%]{2,300}
        )
    (?P=quote)
    """,
    re.VERBOSE,
)

# fetch("/api/..."), axios.get(`...`), $.ajax({url: "..."})
CALL_PATTERN = re.compile(
    r"""(?P<fn>
            fetch
            | axios(?:\.(?:get|post|put|patch|delete|request))?
            | \$\.(?:ajax|get|post|getJSON)
            | XMLHttpRequest
            | open
        )
        \s*
        (?:
            \(\s*(?P<q1>['"`])(?P<url1>[^'"`]{3,400})(?P=q1)
            |
            \(\s*\{[^}]{0,800}?url\s*:\s*(?P<q2>['"`])(?P<url2>[^'"`]{3,400})(?P=q2)
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "baseURL": "https://api.example.com", endpoint: "/videos/search"
KEYED_URL_PATTERN = re.compile(
    r"""(?P<key>
            (?:base)?url
            | endpoint
            | path
            | href
            | uri
            | host
            | api(?:Url|Base|Path|Host)?
            | search(?:Url|Path|Api)?
            | video(?:s)?(?:Url|Path|Api|Search)?
        )
        \s*[:=]\s*
        (?P<q>['"`])(?P<url>[^'"`]{3,400})(?P=q)
    """,
    re.IGNORECASE | re.VERBOSE,
)

GRAPHQL_PATTERN = re.compile(
    r"""(?P<q>['"`])
        (?P<body>
            (?:query|mutation)\s+[A-Za-z0-9_]+\s*[({][^'"`]{10,800}
        )
    (?P=q)
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class Finding:
    url: str
    source: str
    kind: str
    score: int
    reasons: tuple[str, ...]
    context: str


def normalize_url(page_url: str, raw: str) -> str | None:
    raw = raw.strip()
    if not raw or raw.startswith(("javascript:", "data:", "mailto:", "#")):
        return None
    if raw.startswith("//"):
        raw = f"{urlparse(page_url).scheme}:{raw}"
    if raw.startswith(("http://", "https://", "/")):
        return urljoin(page_url, raw)
    # relative API-looking path without leading slash
    if re.match(r"^(api|graphql|gql|v\d+)/", raw, re.I):
        return urljoin(page_url, "/" + raw)
    return None


def looks_like_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(
        (
            ".css",
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".tsx",
            ".jsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".map",
            ".mp4",
            ".webm",
            ".mp3",
            ".html",
            ".htm",
        )
    )


def score_candidate(url: str, context: str) -> tuple[int, list[str]]:
    """Score mostly by URL path; context only adds mild bonus."""
    url_l = url.lower()
    ctx_l = context.lower()
    score = 0
    reasons: list[str] = []

    if looks_like_asset(url):
        return -10, ["static asset"]

    if any(h in url_l for h in API_HINTS):
        score += 3
        reasons.append("api-like")
    elif any(h in ctx_l for h in API_HINTS):
        score += 1
        reasons.append("api-like-context")

    if any(h in url_l for h in VIDEO_HINTS):
        score += 5
        reasons.append("video-related")
    elif any(h in ctx_l for h in VIDEO_HINTS):
        score += 2
        reasons.append("video-context")

    if any(h in url_l for h in SEARCH_HINTS):
        score += 5
        reasons.append("search-related")
    elif any(h in ctx_l for h in SEARCH_HINTS):
        score += 2
        reasons.append("search-context")

    if "graphql" in url_l or "/gql" in url_l:
        score += 3
        reasons.append("graphql")
    if re.search(r"/api(/|$)", url_l):
        score += 2
        reasons.append("/api path")
    if re.search(r"\.(json|xml)(\?|$)", url_l):
        score += 1
        reasons.append("data response")

    # Drop weak matches that are only from nearby comments/code context
    if score < 3:
        return score, reasons
    if not any(
        r in reasons
        for r in ("api-like", "video-related", "search-related", "graphql", "/api path")
    ):
        return 0, reasons

    return score, reasons


def snippet(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    chunk = text[left:right].replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", chunk).strip()


def extract_from_text(text: str, page_url: str, source: str) -> list[Finding]:
    findings: list[Finding] = []
    seen_local: set[tuple[str, str, str]] = set()

    def add(raw_url: str, kind: str, start: int, end: int) -> None:
        full = normalize_url(page_url, raw_url)
        if not full:
            return
        ctx = snippet(text, start, end)
        score, reasons = score_candidate(full, ctx)
        if score < 3:
            return
        key = (full, kind, source)
        if key in seen_local:
            return
        seen_local.add(key)
        findings.append(
            Finding(
                url=full,
                source=source,
                kind=kind,
                score=score,
                reasons=tuple(reasons),
                context=ctx,
            )
        )

    for match in CALL_PATTERN.finditer(text):
        raw = match.group("url1") or match.group("url2")
        if raw:
            add(raw, f"call:{match.group('fn')}", match.start(), match.end())

    for match in KEYED_URL_PATTERN.finditer(text):
        add(match.group("url"), f"key:{match.group('key')}", match.start(), match.end())

    for match in URL_PATTERN.finditer(text):
        add(match.group("url"), "string-url", match.start(), match.end())

    for match in GRAPHQL_PATTERN.finditer(text):
        body = match.group("body")
        lower = body.lower()
        if any(h in lower for h in VIDEO_HINTS + SEARCH_HINTS):
            findings.append(
                Finding(
                    url="(graphql operation in source)",
                    source=source,
                    kind="graphql-query",
                    score=8,
                    reasons=("graphql", "video/search keywords"),
                    context=snippet(text, match.start(), match.end(), radius=120),
                )
            )

    return findings


def fetch_text(session: requests.Session, url: str) -> str | None:
    try:
        response = session.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_JS_BYTES:
            content = content[:MAX_JS_BYTES]
        return content.decode(response.encoding or "utf-8", errors="replace")
    except requests.RequestException as exc:
        print(f"[warn] failed to fetch {url}: {exc}", file=sys.stderr)
        return None


def collect_script_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    urls: list[str] = []
    for tag in soup.find_all("script", src=True):
        src = tag.get("src")
        if not src:
            continue
        full = urljoin(page_url, src)
        if urlparse(full).scheme in ("http", "https"):
            urls.append(full)
    # de-dupe, keep order
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered[:MAX_JS_FILES]


def analyze_page(page_url: str) -> list[Finding]:
    session = requests.Session()
    session.headers.update(HEADERS)

    html = fetch_text(session, page_url)
    if html is None:
        raise SystemExit(f"Cannot load page: {page_url}")

    soup = BeautifulSoup(html, "html.parser")
    findings = extract_from_text(html, page_url, source=page_url)

    for i, script in enumerate(soup.find_all("script"), start=1):
        if script.get("src"):
            continue
        inline = script.string or script.get_text() or ""
        if len(inline.strip()) < 20:
            continue
        findings.extend(
            extract_from_text(inline, page_url, source=f"{page_url}#inline-script-{i}")
        )

    for js_url in collect_script_urls(soup, page_url):
        js_text = fetch_text(session, js_url)
        if not js_text:
            continue
        findings.extend(extract_from_text(js_text, page_url, source=js_url))

    # merge duplicates, keep best score / richest reasons
    best: dict[tuple[str, str], Finding] = {}
    for item in findings:
        key = (item.url, item.kind)
        prev = best.get(key)
        if prev is None or item.score > prev.score or (
            item.score == prev.score and len(item.context) > len(prev.context)
        ):
            best[key] = item

    return sorted(best.values(), key=lambda x: (-x.score, x.url))


def filter_video_search(findings: Iterable[Finding], min_score: int) -> list[Finding]:
    result: list[Finding] = []
    for item in findings:
        if item.score < min_score:
            continue
        reasons = set(item.reasons)
        # Prefer things tied to video and/or search APIs
        if (
            "video-related" in reasons
            or "search-related" in reasons
            or "video-context" in reasons
            or "search-context" in reasons
            or item.kind == "graphql-query"
        ):
            result.append(item)
        elif "api-like" in reasons and item.score >= min_score + 2:
            result.append(item)
    return result


def dedupe_by_url(findings: list[Finding]) -> list[Finding]:
    best: dict[str, Finding] = {}
    for item in findings:
        prev = best.get(item.url)
        if prev is None or item.score > prev.score:
            best[item.url] = item
    return sorted(best.values(), key=lambda x: (-x.score, x.url))


def print_report(findings: list[Finding]) -> None:
    if not findings:
        print("Ничего похожего на video/search API не найдено в HTML/JS.")
        print("Подсказка: часть API грузится только после действий в браузере — смотрите Network.")
        return

    print(f"Найдено кандидатов: {len(findings)}\n")
    reason_counter: Counter[str] = Counter()
    for item in findings:
        reason_counter.update(item.reasons)
        print(f"[{item.score}] {item.url}")
        print(f"  kind:    {item.kind}")
        print(f"  source:  {item.source}")
        print(f"  reasons: {', '.join(item.reasons)}")
        print(f"  context: {item.context}")
        print()

    print("Частые признаки:")
    for reason, count in reason_counter.most_common(8):
        print(f"  {reason}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сканирует HTML и подключённые JS-файлы страницы и ищет "
            "обращения к API, особенно связанные с поиском видео."
        )
    )
    parser.add_argument("url", help="URL страницы для анализа")
    parser.add_argument(
        "--min-score",
        type=int,
        default=5,
        help="Минимальный score кандидата (по умолчанию 5)",
    )
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="Показать все API-подобные URL, не только video/search",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Вывести результат в JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    findings = analyze_page(args.url)
    if not args.all_api:
        findings = filter_video_search(findings, min_score=args.min_score)
    else:
        findings = [f for f in findings if f.score >= args.min_score]
    findings = dedupe_by_url(findings)

    if args.as_json:
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
    else:
        print_report(findings)


if __name__ == "__main__":
    main()
