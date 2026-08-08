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


def log(msg: str) -> None:
    """Progress logs always go to stderr so --json stdout stays clean."""
    print(msg, file=sys.stderr, flush=True)


def log_step(msg: str) -> None:
    log(f"→ {msg}")


def log_ok(msg: str) -> None:
    log(f"✓ Успех: {msg}")


def log_fail(msg: str) -> None:
    log(f"✗ Ошибка: {msg}")


def log_info(msg: str) -> None:
    log(f"• {msg}")


def normalize_url(page_url: str, raw: str) -> str | None:
    raw = raw.strip()
    if not raw or raw.startswith(("javascript:", "data:", "mailto:", "#")):
        return None
    if raw.startswith("//"):
        raw = f"{urlparse(page_url).scheme}:{raw}"
    if raw.startswith(("http://", "https://", "/")):
        return urljoin(page_url, raw)
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


def fetch_text(session: requests.Session, url: str, label: str) -> str | None:
    log_step(f"Загружаю {label}: {url}")
    try:
        response = session.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        content = response.content
        truncated = False
        if len(content) > MAX_JS_BYTES:
            content = content[:MAX_JS_BYTES]
            truncated = True
        text = content.decode(response.encoding or "utf-8", errors="replace")
        size_kb = len(content) / 1024
        extra = " (обрезано по лимиту)" if truncated else ""
        log_ok(f"{label} загружен — HTTP {response.status_code}, {size_kb:.1f} KB{extra}")
        return text
    except requests.Timeout:
        log_fail(f"{label} — таймаут ({DEFAULT_TIMEOUT}s): {url}")
        return None
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        log_fail(f"{label} — HTTP {status}: {url}")
        return None
    except requests.RequestException as exc:
        log_fail(f"{label} — сеть/запрос: {exc}")
        return None


def collect_script_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    log_step("Ищу внешние <script src=...> на странице")
    urls: list[str] = []
    for tag in soup.find_all("script", src=True):
        src = tag.get("src")
        if not src:
            continue
        full = urljoin(page_url, src)
        if urlparse(full).scheme in ("http", "https"):
            urls.append(full)

    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)

    limited = ordered[:MAX_JS_FILES]
    if not limited:
        log_info("Внешние JS-файлы не найдены")
    else:
        log_ok(f"Найдено внешних JS: {len(ordered)}" + (
            f" (беру первые {MAX_JS_FILES})" if len(ordered) > MAX_JS_FILES else ""
        ))
        for i, url in enumerate(limited, start=1):
            log_info(f"JS [{i}/{len(limited)}]: {url}")
    return limited


def analyze_page(page_url: str) -> list[Finding]:
    log("=" * 60)
    log_step(f"Старт анализа страницы: {page_url}")
    log("=" * 60)

    session = requests.Session()
    session.headers.update(HEADERS)

    html = fetch_text(session, page_url, label="HTML страницы")
    if html is None:
        log_fail("Не удалось загрузить страницу — анализ остановлен")
        raise SystemExit(1)

    log_step("Парсю HTML")
    try:
        soup = BeautifulSoup(html, "html.parser")
        log_ok("HTML успешно разобран")
    except Exception as exc:  # noqa: BLE001 - surface parse issues to user
        log_fail(f"Не удалось разобрать HTML: {exc}")
        raise SystemExit(1) from exc

    findings: list[Finding] = []

    log_step("Сканирую HTML на API-ссылки")
    html_hits = extract_from_text(html, page_url, source=page_url)
    findings.extend(html_hits)
    if html_hits:
        log_ok(f"В HTML найдено кандидатов: {len(html_hits)}")
    else:
        log_info("В HTML кандидатов не найдено")

    inline_scripts = [
        (i, script)
        for i, script in enumerate(soup.find_all("script"), start=1)
        if not script.get("src")
    ]
    log_step(f"Проверяю inline-скрипты (всего тегов без src: {len(inline_scripts)})")
    inline_scanned = 0
    inline_hits_total = 0
    for i, script in inline_scripts:
        inline = script.string or script.get_text() or ""
        if len(inline.strip()) < 20:
            continue
        inline_scanned += 1
        log_step(f"Сканирую inline-скрипт #{i} ({len(inline)} символов)")
        hits = extract_from_text(inline, page_url, source=f"{page_url}#inline-script-{i}")
        findings.extend(hits)
        if hits:
            log_ok(f"Inline #{i}: найдено кандидатов — {len(hits)}")
            inline_hits_total += len(hits)
        else:
            log_info(f"Inline #{i}: кандидатов нет")

    if inline_scanned == 0:
        log_info("Подходящих inline-скриптов нет")
    else:
        log_ok(
            f"Inline-скрипты обработаны: {inline_scanned}, "
            f"кандидатов суммарно: {inline_hits_total}"
        )

    js_urls = collect_script_urls(soup, page_url)
    js_ok = 0
    js_fail = 0
    js_hits_total = 0
    for idx, js_url in enumerate(js_urls, start=1):
        js_text = fetch_text(session, js_url, label=f"JS [{idx}/{len(js_urls)}]")
        if not js_text:
            js_fail += 1
            continue
        js_ok += 1
        log_step(f"Сканирую JS [{idx}/{len(js_urls)}]")
        hits = extract_from_text(js_text, page_url, source=js_url)
        findings.extend(hits)
        if hits:
            log_ok(f"JS [{idx}/{len(js_urls)}]: найдено кандидатов — {len(hits)}")
            js_hits_total += len(hits)
        else:
            log_info(f"JS [{idx}/{len(js_urls)}]: кандидатов нет")

    if js_urls:
        log_info(f"JS загружено успешно: {js_ok}, с ошибкой: {js_fail}")
        log_info(f"Кандидатов из внешних JS: {js_hits_total}")

    log_step("Объединяю и ранжирую результаты")
    best: dict[tuple[str, str], Finding] = {}
    for item in findings:
        key = (item.url, item.kind)
        prev = best.get(key)
        if prev is None or item.score > prev.score or (
            item.score == prev.score and len(item.context) > len(prev.context)
        ):
            best[key] = item

    merged = sorted(best.values(), key=lambda x: (-x.score, x.url))
    log_ok(f"После объединения уникальных записей: {len(merged)}")
    return merged


def filter_video_search(findings: Iterable[Finding], min_score: int) -> list[Finding]:
    result: list[Finding] = []
    for item in findings:
        if item.score < min_score:
            continue
        reasons = set(item.reasons)
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
    print()
    print("=" * 60)
    if not findings:
        print("Итог: ничего похожего на video/search API не найдено.")
        print("Подсказка: часть API грузится только в браузере — смотрите Network.")
        print("=" * 60)
        return

    print(f"Итог: найдено кандидатов — {len(findings)}")
    print("=" * 60)
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
    print("=" * 60)


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
        help="Вывести результат в JSON (прогресс всё равно в терминал)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_step("Запуск find_video_api")
    log_info(f"Цель: {args.url}")
    log_info(f"Режим: {'все API' if args.all_api else 'video/search'}, min-score={args.min_score}")

    findings = analyze_page(args.url)

    log_step("Фильтрую кандидатов")
    before = len(findings)
    if not args.all_api:
        findings = filter_video_search(findings, min_score=args.min_score)
    else:
        findings = [f for f in findings if f.score >= args.min_score]
    findings = dedupe_by_url(findings)
    log_ok(f"Фильтрация завершена: было {before}, осталось {len(findings)}")

    if findings:
        log_ok(f"Анализ завершён успешно — найдено {len(findings)} API-кандидатов")
    else:
        log_info("Анализ завершён: подходящих API не найдено")

    if args.as_json:
        log_step("Печатаю JSON-результат в stdout")
        print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=2))
        log_ok("JSON выведен")
    else:
        print_report(findings)


if __name__ == "__main__":
    main()
