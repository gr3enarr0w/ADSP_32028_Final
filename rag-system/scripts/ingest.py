#!/usr/bin/env python3
"""
CLI for ingesting into Efficient Hermes RAG (Qdrant).
Supports local paths (files/dirs), URLs, and APIs.
Usage examples in SKILL.md.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent to path for utils
sys.path.append(str(Path(__file__).parent))

from utils import (
    EfficientRAG, load_config,
    iter_filtered_files, DEFAULT_EXCLUDE_DIRS, DEFAULT_INCLUDE_EXTENSIONS,
    DEFAULT_MAX_FILE_SIZE_MB, WORK_TOPIC_EXCLUDE_DIRS, WORK_TOPIC_EXCLUDE_FILE_PATTERNS,
)


def _run_dry_run(root_path, exclude_dirs, include_extensions, max_file_size_mb,
                  exclude_patterns=None, require_git_authorship_email=None):
    """Walk `root_path` applying the same exclude-dir/include-extension/
    max-size/exclude-pattern/git-authorship filters the real --recursive
    ingest path uses (via utils.iter_filtered_files), WITHOUT touching
    Qdrant or loading any embedding model. Prints total file count, total
    size, and a per-extension breakdown so an operator can sanity-check
    scope/cost before committing to a real (potentially multi-hour)
    ingestion run.
    """
    root = Path(root_path).expanduser().resolve()
    print(f"=== DRY RUN: {root} ===")
    print(f"exclude_dirs: {exclude_dirs}")
    print(f"include_extensions: {include_extensions}")
    print(f"max_file_size_mb: {max_file_size_mb if max_file_size_mb else 'disabled'}")
    print(f"exclude_patterns: {exclude_patterns if exclude_patterns is not None else '(disabled)'}")
    print(f"require_git_authorship_email: {require_git_authorship_email or '(disabled)'}")

    total_files = 0
    total_bytes = 0
    by_ext = {}

    for fpath in iter_filtered_files(
        str(root), exclude_dirs=exclude_dirs,
        include_extensions=include_extensions,
        max_file_size_mb=max_file_size_mb,
        exclude_patterns=exclude_patterns,
        require_git_authorship_email=require_git_authorship_email,
    ):
        try:
            size = os.path.getsize(fpath)
        except OSError:
            continue
        ext = os.path.splitext(fpath)[1].lower() or "(no extension)"
        total_files += 1
        total_bytes += size
        entry = by_ext.setdefault(ext, {"count": 0, "bytes": 0})
        entry["count"] += 1
        entry["bytes"] += size

    print(f"\nTotal files that WOULD be ingested: {total_files}")
    print(f"Total size: {total_bytes / (1024 ** 2):.1f} MB ({total_bytes / (1024 ** 3):.2f} GB)")
    print("\nBreakdown by extension:")
    for ext, stats in sorted(by_ext.items(), key=lambda kv: kv[1]["bytes"], reverse=True):
        print(f"  {ext:>15}  {stats['count']:>8} files  {stats['bytes'] / (1024 ** 2):>10.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Efficient RAG Ingestion for Hermes + Qdrant")
    parser.add_argument("--path", help="Local file or directory path")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories")
    parser.add_argument(
        "--exclude-dirs",
        default=",".join(DEFAULT_EXCLUDE_DIRS),
        help=(
            "Comma-separated directory names to prune anywhere in the path during "
            f"--recursive local ingestion (default: {','.join(DEFAULT_EXCLUDE_DIRS)}). "
            "Directories named *.dist-info/*.egg-info are always excluded in addition "
            "to this list, since they're named after a package+version rather than a "
            "fixed name."
        ),
    )
    parser.add_argument(
        "--include-extensions",
        default=",".join(DEFAULT_INCLUDE_EXTENSIONS),
        help=(
            "Comma-separated file extensions (with leading dot) to treat as ingestible "
            f"content during --recursive local ingestion (default: "
            f"{','.join(DEFAULT_INCLUDE_EXTENSIONS)}). This is an ALLOW-list — anything "
            "not listed (binaries, images, archives, .pyc/.pyi/.so/.dat, etc.) is skipped."
        ),
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE_MB,
        help=(
            f"Skip files larger than this many MB even if their extension is included "
            f"(default: {DEFAULT_MAX_FILE_SIZE_MB}) — guards against large data dumps "
            "(e.g. a multi-GB .csv/.xlsx export) being chunked/embedded as prose. "
            "Pass 0 or a negative value to disable the size guard."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "For --path --recursive only: walk and apply filters WITHOUT ingesting "
            "anything (no Qdrant writes, no embedding). Prints file count, total size, "
            "and a per-extension breakdown, then exits."
        ),
    )
    parser.add_argument(
        "--require-git-authorship",
        metavar="EMAIL",
        default=None,
        help=(
            "For --path --recursive only: for each TOP-LEVEL directory under --path "
            "that is itself a git repository (has a .git directory), require at least "
            "one commit authored by EMAIL (checked via `git log --author=EMAIL "
            "--oneline -1` inside it) or exclude that ENTIRE top-level directory from "
            "ingestion. Top-level directories that are NOT git repositories at all are "
            "completely unaffected by this flag — it only applies to things that ARE "
            "git repos. Off by default (None), so this never silently changes behavior "
            "for callers/other directories (e.g. ~/Downloads, ~/Documents) that don't "
            "pass it."
        ),
    )
    parser.add_argument(
        "--exclude-patterns",
        default=None,
        help=(
            "Comma-separated case-insensitive substrings; any FILE (not directory) "
            "whose bare name contains one is skipped during --recursive local "
            "ingestion, on top of the normal extension/size/exclude-dir filters. "
            "Off by default (None) — no substring-pattern filtering at all — so this "
            "never silently changes behavior for callers who don't pass it. NOTE: "
            "`.env` / `.env.*` files are excluded UNCONDITIONALLY regardless of this "
            "flag (a standing secrets guard applied directly in "
            "utils.iter_filtered_files, independent of --exclude-patterns and of "
            "--work-topic-excludes) — passing or omitting this flag has no effect on "
            "that guard."
        ),
    )
    parser.add_argument(
        "--work-topic-excludes",
        action="store_true",
        help=(
            "Convenience flag: merges utils.WORK_TOPIC_EXCLUDE_DIRS into --exclude-dirs "
            "and utils.WORK_TOPIC_EXCLUDE_FILE_PATTERNS into --exclude-patterns, AND "
            "enables EfficientRAG.ingest()'s exclude_work_content check (scans each "
            "file's EXTRACTED TEXT, not just its name/path, for any "
            "utils.WORK_CONTENT_SIGNAL_TERMS substring and skips ingesting it entirely "
            "if found — added after filename/path-only filtering missed real "
            "contaminated files with generic names like `Unknown.pdf` in a live "
            "~/Downloads ingestion run; see docs/CHANGELOG.md). All three lists name "
            "THIS USER's own confirmed Example Organization/Third-Party Service work content (found via manual "
            "inspection) — NOT a generic default for other users/callers of this skill, "
            "which is why it's opt-in via this single flag rather than folded into the "
            "relevant defaults. Off by default. NOTE: the `secrets/` directory guard and "
            "the `.env`/`.env.*` file guard are separate, unconditional security checks "
            "applied directly in utils.py regardless of whether this flag is passed — "
            "turning this flag off (e.g. for a personal-only corpus that still wants "
            "real work content included) does NOT disable either of those."
        ),
    )
    parser.add_argument("--url", help="Web URL to ingest")
    parser.add_argument(
        "--crawl-url",
        help=(
            "Hub/index URL to crawl (breadth-first, same-domain + same-path-prefix "
            "by default) and ingest EVERY discovered linked page from, rather than "
            "just the index page itself. Use for doc sites where the start URL is "
            "just a list of links out to individual articles (see utils.EfficientRAG."
            "crawl_site() for scope/dedup rules)."
        ),
    )
    parser.add_argument(
        "--crawl-max-pages", type=int, default=200,
        help="Max unique pages to fetch for --crawl-url (default: 200).",
    )
    parser.add_argument(
        "--no-crawl-same-prefix", dest="crawl_same_prefix", action="store_false",
        help=(
            "For --crawl-url: disable the default same-URL-path-prefix scope "
            "restriction, following any link on the SAME DOMAIN as --crawl-url "
            "regardless of path. Use for a genuine cross-section site index page "
            "(e.g. an 'all documentation' hub linking out to many differently-"
            "pathed doc spaces on the same domain) where path-prefix scoping "
            "would wrongly limit the crawl to just the index page itself. "
            "Default: prefix-scoped (see utils.EfficientRAG.crawl_site())."
        ),
    )
    parser.set_defaults(crawl_same_prefix=True)
    parser.add_argument(
        "--crawl-path-prefix",
        help=(
            "For --crawl-url: override the auto-derived same-path-prefix scope with an "
            "exact prefix (e.g. '/docs/rest-api/'). Use when --crawl-url's LAST path "
            "segment is itself the section root (children live directly under it, e.g. "
            "'/docs/rest-api' -> '/docs/rest-api/getting-started') rather than a hub page "
            "one level below the section — the auto-derivation (drop the last segment) "
            "assumes the latter and would otherwise wander into every unrelated doc "
            "section on the domain (see utils.EfficientRAG.crawl_site() and "
            "docs/CHANGELOG.md 2026-07-22 Fivetran entry). No effect with "
            "--no-crawl-same-prefix."
        ),
    )
    parser.add_argument("--api", help="API endpoint URL")
    parser.add_argument("--method", default="GET", help="HTTP method for API")
    parser.add_argument("--headers", help="JSON string of headers for API")
    parser.add_argument("--gdrive-query", help="Search Google Drive (full-text) and ingest all matching Docs/Sheets")
    parser.add_argument("--gdrive-max-results", type=int, default=10, help="Max files to ingest from --gdrive-query")
    parser.add_argument(
        "--google-involvement",
        choices=["owner_only", "owner_or_writer", "owner_writer_or_commented", "any"],
        default="owner_or_writer",
        help=(
            "Filter --gdrive-query results to files the user is actually involved with, not "
            "just able to see (default: owner_or_writer). 'owner_only' = 'me' in owners only. "
            "'owner_or_writer' = 'me' in owners or 'me' in writers (default). "
            "'owner_writer_or_commented' = owner/writer PLUS a supplementary comments.list() "
            "check on other visible files (heavier: one extra API call per extra candidate "
            "file) to also catch files the user commented on but doesn't own/edit. "
            "'any' = no involvement filter (original, visibility-only behavior)."
        ),
    )
    parser.add_argument("--gdoc-id", help="Ingest a specific Google Doc by file ID")
    parser.add_argument("--gsheet-id", help="Ingest a specific Google Sheet by file ID")
    parser.add_argument("--gslide-id", help="Ingest a specific Google Slides presentation by file ID")
    parser.add_argument(
        "--gsheet-range", default=None,
        help=(
            "A1-notation range for --gsheet-id / sheets found via --gdrive-query. "
            "If omitted (default), the actual tab/sheet name is looked up via the "
            "Sheets API (spreadsheets.get) and the first real tab is used, rather "
            "than assuming the first tab is literally named 'Sheet1' (which is not "
            "true for every spreadsheet and causes a 400 'Unable to parse range' "
            "error if hardcoded). If explicitly provided, it is honored as-is for "
            "every sheet ingested in this run."
        ),
    )
    parser.add_argument("--google-client-secret", help="Override google_workspace.client_secret_path from config.yaml")
    parser.add_argument("--google-token-path", help="Override google_workspace.token_path from config.yaml")
    parser.add_argument(
        "--confluence-dump-dir",
        help=(
            "Ingest Document Service Cloud pages from pre-fetched JSON dump files (one file "
            "per space, e.g. produced via the Third-Party Service MCP tools — "
            "mcp__jira_prod__getPagesInDocument ServiceSpace / getDocument ServicePage — rather "
            "than a standalone API-token HTTP client; a prior approach requiring a "
            "Document Service-scoped Third-Party Service API token was removed in favor of this because "
            "the MCP tools are already authenticated in-session, see docs/CHANGELOG.md "
            "2026-07-21). Every *.json file directly inside this directory is read and "
            "expected to contain a JSON array of page objects, each with at minimum "
            "'page_id', 'title', 'url' (the real citable webui URL), and 'text' (the "
            "extracted body content). Pages with empty/whitespace-only 'text' are "
            "skipped. Filename is not semantically significant (space key is read from "
            "each page object's own 'space' field if present, else falls back to the "
            "file's basename without extension)."
        ),
    )
    parser.add_argument(
        "--collection",
        required=False,
        help="Target collection name (required unless --dry-run)",
    )
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--mode", choices=["light", "balanced", "full"], default="light",
                        help="Indexing mode (light = max efficiency)")
    parser.add_argument("--force-recreate", action="store_true", help="Delete and recreate collection")
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args()

    exclude_dirs = [d.strip() for d in args.exclude_dirs.split(",") if d.strip()]
    include_extensions = [e.strip().lower() for e in args.include_extensions.split(",") if e.strip()]
    max_file_size_mb = args.max_file_size_mb if args.max_file_size_mb and args.max_file_size_mb > 0 else None

    exclude_patterns = None
    if args.exclude_patterns is not None:
        exclude_patterns = [p.strip() for p in args.exclude_patterns.split(",") if p.strip()]

    if args.work_topic_excludes:
        # De-duplicate while preserving order (dict.fromkeys) in case the user's
        # own --exclude-dirs/--exclude-patterns already overlapped with these.
        exclude_dirs = list(dict.fromkeys(exclude_dirs + WORK_TOPIC_EXCLUDE_DIRS))
        base_patterns = exclude_patterns if exclude_patterns is not None else []
        exclude_patterns = list(dict.fromkeys(base_patterns + WORK_TOPIC_EXCLUDE_FILE_PATTERNS))

    require_git_authorship_email = args.require_git_authorship

    if args.dry_run:
        if not (args.path and args.recursive):
            print("--dry-run currently only supports --path with --recursive")
            sys.exit(1)
        _run_dry_run(
            args.path, exclude_dirs, include_extensions, max_file_size_mb,
            exclude_patterns=exclude_patterns,
            require_git_authorship_email=require_git_authorship_email,
        )
        return

    if not args.collection:
        print("--collection is required (unless --dry-run)")
        sys.exit(1)

    config = load_config(args.config)
    config["indexing"]["mode"] = args.mode

    rag = EfficientRAG(config)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    stats_list = []

    if args.path:
        path = Path(args.path).expanduser().resolve()
        if path.is_dir():
            if args.recursive:
                files = [Path(p) for p in iter_filtered_files(
                    str(path), exclude_dirs=exclude_dirs,
                    include_extensions=include_extensions,
                    max_file_size_mb=max_file_size_mb,
                    exclude_patterns=exclude_patterns,
                    require_git_authorship_email=require_git_authorship_email,
                )]
            else:
                files = list(path.iterdir())
            recreate_pending = args.force_recreate
            for f in files:
                if f.is_file() and (not args.recursive) and f.suffix.lower() not in [".pdf", ".txt", ".md", ".docx", ".html"]:
                    # Non-recursive single-directory listing keeps the original,
                    # narrower default extension set for backward compatibility.
                    # --recursive already applied include_extensions filtering
                    # above via iter_filtered_files, so every entry in `files`
                    # there is already eligible.
                    continue
                if f.is_file():
                    print(f"Ingesting file: {f}")
                    # Only recreate the collection before the FIRST file in a
                    # directory ingest — recreating on every file would wipe
                    # out every previously-ingested file's points, leaving
                    # only the last file's chunks in the collection.
                    try:
                        stat = rag.ingest(str(f), args.collection, tags=tags, force_recreate=recreate_pending, exclude_work_content=args.work_topic_excludes)
                        stats_list.append(stat)
                    except Exception as exc:
                        # Defense in depth: a single file's loader/chunking/
                        # embedding failure (any file-type-specific bug, not
                        # just the PDF one this guard was added for) must
                        # never abort the entire --recursive batch — see
                        # docs/CHANGELOG.md 2026-07-22. Log and continue,
                        # matching the resilience policy already used by the
                        # --gdrive-query and --confluence-dump-dir loops below.
                        print(f"ERROR ingesting {f}: {exc!r} — skipped, continuing")
                    recreate_pending = False
        elif path.is_file():
            stat = rag.ingest(str(path), args.collection, tags=tags, force_recreate=args.force_recreate, exclude_work_content=args.work_topic_excludes)
            stats_list.append(stat)
    elif args.url:
        stat = rag.ingest(args.url, args.collection, tags=tags, is_url=True, force_recreate=args.force_recreate, exclude_work_content=args.work_topic_excludes)
        stats_list.append(stat)
    elif args.crawl_url:
        def _ingest_crawled_page(url, text, recreate):
            extra_metadata = {
                "source_type": "external_docs",
                "url": url,
            }
            return rag.ingest(
                url, args.collection, tags=tags,
                raw_text=text, extra_metadata=extra_metadata, force_recreate=recreate,
                exclude_work_content=args.work_topic_excludes,
            )

        print(f"Crawling {args.crawl_url} (max_pages={args.crawl_max_pages})")
        pages = rag.crawl_site(
            args.crawl_url, max_pages=args.crawl_max_pages, same_prefix=args.crawl_same_prefix,
            path_prefix=args.crawl_path_prefix,
        )
        print(f"Crawl discovered {len(pages)} page(s) with extractable text")

        recreate_pending = args.force_recreate
        for page in pages:
            url = page["url"]
            text = page.get("text", "") or ""
            title = page.get("title", url)
            if not text.strip():
                print(f"Skipping empty crawled page: {title!r} ({url})")
                continue
            try:
                print(f"Ingesting crawled page: {title!r} ({url})")
                stat = _ingest_crawled_page(url, text, recreate_pending)
                recreate_pending = False
                stats_list.append(stat)
            except Exception as exc:
                # A single page's ingest failure must not abort the whole
                # crawl (same resilience policy as the --gdrive-query and
                # --confluence-dump-dir batch loops above).
                print(f"ERROR ingesting crawled page {url!r}: {exc!r} — skipped, continuing")
    elif args.api:
        import json
        api_headers = json.loads(args.headers) if args.headers else None
        api_config = {"method": args.method, "headers": api_headers}
        stat = rag.ingest(args.api, args.collection, tags=tags, is_api=True, api_config=api_config, force_recreate=args.force_recreate, exclude_work_content=args.work_topic_excludes)
        stats_list.append(stat)
    elif args.gdrive_query is not None or args.gdoc_id or args.gsheet_id or args.gslide_id:
        from google_workspace import (
            GoogleWorkspaceClient,
            GOOGLE_DOC_MIME_TYPE,
            GOOGLE_SHEET_MIME_TYPE,
            GOOGLE_SLIDES_MIME_TYPE,
            sheet_values_to_text,
        )

        gw_config = config.get("google_workspace", {})
        client_secret_path = args.google_client_secret or gw_config.get("client_secret_path")
        token_path = args.google_token_path or gw_config.get("token_path")
        gw_client = GoogleWorkspaceClient(client_secret_path, token_path)

        def _resolve_gsheet_range(sheet_id, explicit_range):
            """Return the A1-notation range to fetch for `sheet_id`.

            If the caller passed an explicit --gsheet-range, honor it as-is
            (backward-compatible single-file path where the user knows the
            range they want). Otherwise, look up the spreadsheet's REAL
            tab/sheet names via the Sheets API rather than assuming the
            first tab is literally named "Sheet1" — that assumption breaks
            (400 "Unable to parse range: Sheet1") for any spreadsheet whose
            first tab has a different name, e.g. one auto-named after a
            date/timestamp.
            """
            if explicit_range:
                return explicit_range
            titles = gw_client.get_spreadsheet_sheet_titles(sheet_id)
            if not titles:
                raise ValueError(f"Spreadsheet {sheet_id} has no sheets/tabs to ingest")
            return titles[0]

        def _ingest_google_text(text, webviewlink, mime_type, name, recreate):
            extra_metadata = {
                "source_type": "google_drive",
                "webViewLink": webviewlink,
                "mime_type": mime_type,
                "google_name": name,
            }
            return rag.ingest(
                webviewlink or name, args.collection, tags=tags,
                raw_text=text, extra_metadata=extra_metadata, force_recreate=recreate,
                exclude_work_content=args.work_topic_excludes,
            )

        recreate_pending = args.force_recreate

        if args.gdrive_query is not None:
            files = gw_client.search_drive(
                args.gdrive_query,
                max_results=args.gdrive_max_results,
                involvement=args.google_involvement,
            )
            if not files:
                print(f"No Drive files matched query: {args.gdrive_query!r}")
            for f in files:
                mime = f.get("mimeType", "")
                name = f.get("name")
                link = f.get("webViewLink")
                try:
                    if mime == GOOGLE_DOC_MIME_TYPE:
                        print(f"Ingesting Google Doc: {name} ({link})")
                        text = gw_client.get_doc_text(f["id"])
                    elif mime == GOOGLE_SHEET_MIME_TYPE:
                        resolved_range = _resolve_gsheet_range(f["id"], args.gsheet_range)
                        print(f"Ingesting Google Sheet: {name} ({link}, range: {resolved_range})")
                        text = sheet_values_to_text(gw_client.get_sheet_values(f["id"], resolved_range))
                    elif mime == GOOGLE_SLIDES_MIME_TYPE:
                        print(f"Ingesting Google Slides: {name} ({link})")
                        text = gw_client.get_slides_text(f["id"])
                    else:
                        print(f"Skipping unsupported Drive mimeType {mime!r} for {name}")
                        continue
                    stat = _ingest_google_text(text, link, mime, name, recreate_pending)
                    recreate_pending = False
                    stats_list.append(stat)
                except Exception as exc:
                    # A single file's fetch/ingest failure (e.g. a Sheet whose
                    # first tab isn't literally named "Sheet1", so the default
                    # --gsheet-range 400s; a transient API error; etc.) must
                    # not abort the entire --gdrive-query batch. Log and move
                    # on to the next file instead.
                    print(f"ERROR ingesting {name!r} ({link}): {exc!r} — skipped, continuing")

        if args.gdoc_id:
            print(f"Ingesting Google Doc by ID: {args.gdoc_id}")
            text = gw_client.get_doc_text(args.gdoc_id)
            link = f"https://docs.google.com/document/d/{args.gdoc_id}/edit"
            stat = _ingest_google_text(text, link, GOOGLE_DOC_MIME_TYPE, args.gdoc_id, recreate_pending)
            recreate_pending = False
            stats_list.append(stat)

        if args.gsheet_id:
            resolved_range = _resolve_gsheet_range(args.gsheet_id, args.gsheet_range)
            print(f"Ingesting Google Sheet by ID: {args.gsheet_id} (range: {resolved_range})")
            text = sheet_values_to_text(gw_client.get_sheet_values(args.gsheet_id, resolved_range))
            link = f"https://docs.google.com/spreadsheets/d/{args.gsheet_id}/edit"
            stat = _ingest_google_text(text, link, GOOGLE_SHEET_MIME_TYPE, args.gsheet_id, recreate_pending)
            recreate_pending = False
            stats_list.append(stat)

        if args.gslide_id:
            print(f"Ingesting Google Slides by ID: {args.gslide_id}")
            text = gw_client.get_slides_text(args.gslide_id)
            link = f"https://docs.google.com/presentation/d/{args.gslide_id}/edit"
            stat = _ingest_google_text(text, link, GOOGLE_SLIDES_MIME_TYPE, args.gslide_id, recreate_pending)
            recreate_pending = False
            stats_list.append(stat)
    elif args.confluence_dump_dir:
        import json as _json

        def _ingest_confluence_text(text, url, title, space_key, page_id, recreate):
            extra_metadata = {
                "source_type": "confluence",
                "space": space_key,
                "url": url,
                "title": title,
                "confluence_page_id": page_id,
            }
            return rag.ingest(
                url, args.collection, tags=tags,
                raw_text=text, extra_metadata=extra_metadata, force_recreate=recreate,
                exclude_work_content=args.work_topic_excludes,
            )

        dump_dir = Path(args.confluence_dump_dir).expanduser().resolve()
        dump_files = sorted(dump_dir.glob("*.json"))
        if not dump_files:
            print(f"No *.json dump files found in {dump_dir}")
            sys.exit(1)

        recreate_pending = args.force_recreate

        for dump_file in dump_files:
            fallback_space_key = dump_file.stem
            try:
                with open(dump_file) as fh:
                    pages = _json.load(fh)
            except Exception as exc:
                print(f"ERROR reading Document Service dump {dump_file}: {exc!r} — skipped, continuing")
                continue
            print(f"Document Service dump {dump_file.name}: {len(pages)} page(s) found")
            for page in pages:
                page_id = page.get("page_id")
                title = page.get("title", "")
                text = page.get("text", "") or ""
                url = page.get("url", "")
                space_key = page.get("space") or fallback_space_key
                if not text.strip():
                    print(f"Skipping empty Document Service page: {title!r} ({page_id})")
                    continue
                try:
                    print(f"Ingesting Document Service page: {title!r} ({url})")
                    stat = _ingest_confluence_text(text, url, title, space_key, page_id, recreate_pending)
                    recreate_pending = False
                    stats_list.append(stat)
                except Exception as exc:
                    # A single page's ingest failure must not abort the whole
                    # dump directory (same resilience policy as the
                    # --gdrive-query batch loop above).
                    print(f"ERROR ingesting Document Service page {page_id!r}: {exc!r} — skipped, continuing")
    else:
        print("Provide --path, --url, --crawl-url, --api, --gdrive-query, --gdoc-id, --gsheet-id, --gslide-id, or --confluence-dump-dir")
        sys.exit(1)

    print("\n=== Ingestion Complete ===")
    for s in stats_list:
        print(s)


if __name__ == "__main__":
    main()
