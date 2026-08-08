#!/usr/bin/env python3
"""Find API endpoints related to video/search in a page and its JS assets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 20
MAX_JS_FILES = 80
MAX_JS_BYTES = 2_000_000
DEFAULT_BROWSER_WAIT_MS = 4000

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

# "/api/" + "videos/" + "search"
CONCAT_PATTERN = re.compile(
    r"""(?P<body>
            (?:['"`][^'"`]{1,120}['"`]\s*\+\s*){1,12}
            ['"`][^'"`]{1,120}['"`]
        )
    """,
    re.VERBOSE,
)

# import/require/dynamic import of JS modules
JS_REF_PATTERN = re.compile(
    r"""(?:
            import\s*\(\s*(?P<q1>['"`])(?P<u1>[^'"`]+)(?P=q1)\s*\)
            | require\s*\(\s*(?P<q2>['"`])(?P<u2>[^'"`]+)(?P=q2)\s*\)
            | from\s+(?P<q3>['"`])(?P<u3>[^'"`]+)(?P=q3)
            | import\s+(?P<q4>['"`])(?P<u4>[^'"`]+)(?P=q4)
            | (?:src|href)\s*[:=]\s*(?P<q5>['"`])(?P<u5>[^'"`]+\.(?:js|mjs|cjs)(?:\?[^'"`]*)?)(?P=q5)
            | (?P<q6>['"`])(?P<u6>[^'"`]*chunk[^'"`]*\.(?:js|mjs)(?:\?[^'"`]*)?)(?P=q6)
            | sourceMappingURL=(?P<u7>\S+\.map)
        )
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
    raw = raw.strip().strip("'\"")
    if not raw or raw.startswith(("javascript:", "data:", "mailto:", "#", "node:", "webpack:")):
        return None
    if raw.startswith("//"):
        raw = f"{urlparse(page_url).scheme}:{raw}"
    if raw.startswith(("http://", "https://", "/")):
        return urljoin(page_url, raw)
    if re.match(r"^(api|graphql|gql|v\d+)/", raw, re.I):
        return urljoin(page_url, "/" + raw)
    # relative module paths like ./chunk.js or ../api/client.js
    if raw.startswith(("./", "../")) or raw.endswith((".js", ".mjs", ".cjs")):
        return urljoin(page_url, raw)
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


def looks_like_js_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith((".js", ".mjs", ".cjs")):
        return True
    # next/nuxt/vite sometimes omit extension in import paths; keep path-like modules
    if "/_next/" in path or "/assets/" in path or "/static/" in path or "chunk" in path:
        return True
    return False


def score_candidate(url: str, context: str) -> tuple[int, list[str]]:
    url_l = url.lower()
    ctx_l = context.lower()
    path = urlparse(url).path
    score = 0
    reasons: list[str] = []

    # Too generic roots like "/api" or "/api/" are usually noise.
    if re.fullmatch(r"/api/?", path, flags=re.I):
        return 0, ["too-generic"]

    if looks_like_asset(url) and not any(
        h in url_l for h in ("api", "graphql", "search", "video", "media")
    ):
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
    if "runtime-network" in ctx_l:
        score += 2
        reasons.append("seen-in-browser")

    if score < 3:
        return score, reasons
    if not any(
        r in reasons
        for r in (
            "api-like",
            "video-related",
            "search-related",
            "graphql",
            "/api path",
            "seen-in-browser",
        )
    ):
        return 0, reasons

    return score, reasons


def snippet(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    chunk = text[left:right].replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", chunk).strip()


def rebuild_concat(expr: str) -> str | None:
    parts = re.findall(r"""['"`]([^'"`]*)['"`]""", expr)
    if len(parts) < 2:
        return None
    joined = "".join(parts)
    if len(joined) < 4:
        return None
    return joined


def extract_from_text(text: str, page_url: str, source: str) -> list[Finding]:
    findings: list[Finding] = []
    seen_local: set[tuple[str, str, str]] = set()

    def add(raw_url: str, kind: str, start: int, end: int, extra_ctx: str = "") -> None:
        full = normalize_url(page_url, raw_url)
        if not full:
            return
        ctx = snippet(text, start, end)
        if extra_ctx:
            ctx = f"{extra_ctx} | {ctx}"
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

    for match in CONCAT_PATTERN.finditer(text):
        rebuilt = rebuild_concat(match.group("body"))
        if rebuilt and ("/" in rebuilt or "http" in rebuilt.lower()):
            add(
                rebuilt,
                "concat-string",
                match.start(),
                match.end(),
                extra_ctx="rebuilt-from-concatenation",
            )

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


def discover_js_refs(text: str, base_url: str) -> list[str]:
    found: list[str] = []
    for match in JS_REF_PATTERN.finditer(text):
        raw = (
            match.group("u1")
            or match.group("u2")
            or match.group("u3")
            or match.group("u4")
            or match.group("u5")
            or match.group("u6")
            or match.group("u7")
        )
        if not raw:
            continue
        # skip source maps as scan targets; they are huge and optional
        if raw.endswith(".map"):
            continue
        full = normalize_url(base_url, raw)
        if full and looks_like_js_url(full):
            found.append(full)
    return found


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


def collect_html_script_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    log_step("Ищу JS в HTML (<script>, modulepreload, prefetch)")
    urls: list[str] = []

    for tag in soup.find_all("script", src=True):
        src = tag.get("src")
        if src:
            full = urljoin(page_url, src)
            if urlparse(full).scheme in ("http", "https"):
                urls.append(full)

    for tag in soup.find_all("link"):
        rel = " ".join(tag.get("rel") or []).lower()
        as_attr = (tag.get("as") or "").lower()
        href = tag.get("href")
        if not href:
            continue
        if "modulepreload" in rel or "preload" in rel and as_attr in {"script", "worker"}:
            full = urljoin(page_url, href)
            if urlparse(full).scheme in ("http", "https"):
                urls.append(full)
        elif href.endswith((".js", ".mjs", ".cjs")):
            full = urljoin(page_url, href)
            if urlparse(full).scheme in ("http", "https"):
                urls.append(full)

    # unique preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)

    if not ordered:
        log_info("В HTML внешних JS не найдено")
    else:
        log_ok(f"В HTML найдено JS-ссылок: {len(ordered)}")
        for i, url in enumerate(ordered[:20], start=1):
            log_info(f"HTML JS [{i}]: {url}")
        if len(ordered) > 20:
            log_info(f"... и ещё {len(ordered) - 20}")
    return ordered


def merge_findings(findings: list[Finding]) -> list[Finding]:
    best: dict[tuple[str, str], Finding] = {}
    for item in findings:
        key = (item.url, item.kind)
        prev = best.get(key)
        if prev is None or item.score > prev.score or (
            item.score == prev.score and len(item.context) > len(prev.context)
        ):
            best[key] = item
    return sorted(best.values(), key=lambda x: (-x.score, x.url))


def crawl_js_queue(
    session: requests.Session,
    page_url: str,
    seed_urls: list[str],
    findings: list[Finding],
    deep: bool,
) -> None:
    queue: deque[str] = deque(seed_urls)
    seen: set[str] = set()
    js_ok = 0
    js_fail = 0
    js_hits_total = 0
    discovered_extra = 0

    log_step(
        "Начинаю обход JS-файлов"
        + (" (рекурсивно по import/require/chunk)" if deep else " (только прямые ссылки)")
    )

    while queue and len(seen) < MAX_JS_FILES:
        js_url = queue.popleft()
        if js_url in seen:
            continue
        seen.add(js_url)
        idx = len(seen)

        js_text = fetch_text(session, js_url, label=f"JS [{idx}]")
        if not js_text:
            js_fail += 1
            continue
        js_ok += 1

        log_step(f"Сканирую JS [{idx}]: {js_url}")
        hits = extract_from_text(js_text, page_url, source=js_url)
        findings.extend(hits)
        if hits:
            log_ok(f"JS [{idx}]: найдено кандидатов — {len(hits)}")
            js_hits_total += len(hits)
        else:
            log_info(f"JS [{idx}]: кандидатов нет")

        if deep:
            refs = discover_js_refs(js_text, js_url)
            new_refs = [u for u in refs if u not in seen and u not in queue]
            if new_refs:
                discovered_extra += len(new_refs)
                log_ok(f"JS [{idx}]: нашёл ещё связанных файлов — {len(new_refs)}")
                for ref in new_refs:
                    if len(seen) + len(queue) >= MAX_JS_FILES:
                        break
                    queue.append(ref)
                    log_info(f"В очередь: {ref}")

    if seen:
        log_info(f"JS обработано: успех={js_ok}, ошибка={js_fail}, всего={len(seen)}")
        log_info(f"Кандидатов из JS: {js_hits_total}")
        if deep:
            log_info(f"Дополнительно найдено связанных JS: {discovered_extra}")
    else:
        log_info("Нечего обходить: очередь JS пуста")


def capture_with_browser(
    page_url: str,
    wait_ms: int,
) -> tuple[list[str], list[Finding]]:
    log_step("Запускаю браузер (Playwright) для ловли динамических JS и API")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log_fail("Playwright не установлен. Выполни: pip install playwright && playwright install chromium")
        raise SystemExit(1)

    script_urls: list[str] = []
    network_findings: list[Finding] = []
    seen_net: set[str] = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=HEADERS["User-Agent"])
            page = context.new_page()

            def on_request(request) -> None:
                url = request.url
                resource = request.resource_type
                if resource in {"script", "worker"}:
                    if url not in script_urls:
                        script_urls.append(url)
                        log_info(f"Браузер загрузил JS: {url}")
                    return

                if resource not in {"xhr", "fetch"}:
                    return
                if url in seen_net:
                    return
                seen_net.add(url)
                ctx = f"runtime-network method={request.method} type={resource}"
                score, reasons = score_candidate(url, ctx + " video search api")
                # for runtime, keep weaker API-like xhr too
                if score < 3 and any(h in url.lower() for h in API_HINTS + VIDEO_HINTS + SEARCH_HINTS):
                    score = max(score, 4)
                    reasons = list(dict.fromkeys(reasons + ["runtime-api"]))
                if score >= 3:
                    network_findings.append(
                        Finding(
                            url=url,
                            source=f"browser:{page_url}",
                            kind=f"runtime:{resource}",
                            score=score,
                            reasons=tuple(reasons) if reasons else ("seen-in-browser",),
                            context=ctx,
                        )
                    )
                    log_ok(f"Браузер поймал API-запрос: {url}")

            page.on("request", on_request)
            log_step(f"Открываю страницу в браузере: {page_url}")
            page.goto(page_url, wait_until="networkidle", timeout=60000)
            log_ok("Страница открыта, жду дополнительные загрузки")
            page.wait_for_timeout(wait_ms)

            # also collect script tags present after hydration
            for src in page.eval_on_selector_all(
                "script[src]",
                "els => els.map(e => e.src).filter(Boolean)",
            ):
                if src not in script_urls:
                    script_urls.append(src)

            browser.close()
    except Exception as exc:  # noqa: BLE001
        log_fail(f"Браузерный режим упал: {exc}")
        raise SystemExit(1) from exc

    log_ok(f"Браузер: JS-файлов замечено — {len(script_urls)}")
    log_ok(f"Браузер: сетевых API-кандидатов — {len(network_findings)}")
    return script_urls, network_findings


def analyze_page(page_url: str, deep: bool = True, use_browser: bool = False, wait_ms: int = DEFAULT_BROWSER_WAIT_MS) -> list[Finding]:
    log("=" * 60)
    log_step(f"Старт анализа страницы: {page_url}")
    log_info(f"Глубокий обход JS: {'да' if deep else 'нет'}")
    log_info(f"Браузерный режим: {'да' if use_browser else 'нет'}")
    log("=" * 60)

    session = requests.Session()
    session.headers.update(HEADERS)
    findings: list[Finding] = []
    seed_js: list[str] = []

    if use_browser:
        browser_js, network_hits = capture_with_browser(page_url, wait_ms=wait_ms)
        seed_js.extend(browser_js)
        findings.extend(network_hits)

    html = fetch_text(session, page_url, label="HTML страницы")
    if html is None and not seed_js:
        log_fail("Не удалось загрузить страницу — анализ остановлен")
        raise SystemExit(1)

    if html is not None:
        log_step("Парсю HTML")
        try:
            soup = BeautifulSoup(html, "html.parser")
            log_ok("HTML успешно разобран")
        except Exception as exc:  # noqa: BLE001
            log_fail(f"Не удалось разобрать HTML: {exc}")
            raise SystemExit(1) from exc

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

            # nested JS refs from inline too
            for ref in discover_js_refs(inline, page_url):
                if ref not in seed_js:
                    seed_js.append(ref)

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

        seed_js.extend(collect_html_script_urls(soup, page_url))

    # unique seeds
    uniq_seed: list[str] = []
    seen_seed: set[str] = set()
    for url in seed_js:
        if url not in seen_seed:
            seen_seed.add(url)
            uniq_seed.append(url)

    crawl_js_queue(session, page_url, uniq_seed, findings, deep=deep)

    log_step("Объединяю и ранжирую результаты")
    merged = merge_findings(findings)
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
            or "seen-in-browser" in reasons
            or item.kind == "graphql-query"
            or item.kind.startswith("runtime:")
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
        print("Подсказка: попробуй --browser, если API грузится только в рантайме.")
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
            "Сканирует HTML и JS (включая связанные/динамические файлы) "
            "и ищет обращения к API, особенно video/search."
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
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="Не ходить рекурсивно по import/require/chunk из JS",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Открыть страницу в headless-браузере и ловить динамические JS/API",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=DEFAULT_BROWSER_WAIT_MS,
        help="Сколько ждать после загрузки в --browser (мс)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deep = not args.no_deep

    log_step("Запуск find_video_api")
    log_info(f"Цель: {args.url}")
    log_info(
        f"Режим: {'все API' if args.all_api else 'video/search'}, "
        f"min-score={args.min_score}, deep={deep}, browser={args.browser}"
    )

    findings = analyze_page(
        args.url,
        deep=deep,
        use_browser=args.browser,
        wait_ms=args.wait_ms,
    )

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
