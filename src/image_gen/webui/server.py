from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import TextIO

from image_gen.runtime_options import (
    add_runtime_startup_arguments,
    argv_for_primary_parser,
    bootstrap_runtime_startup,
)
from image_gen.webui.diagnostics import write_webui_failure_bundle
from image_gen.webui.store import WebUIStore
from modules.project_context import ProjectContext


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_PORT_SEARCH_LIMIT = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local IMAGE_GEN WebUI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--port-search-limit",
        type=int,
        default=DEFAULT_PORT_SEARCH_LIMIT,
        help="Number of consecutive ports to try when the requested port is occupied.",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail instead of advancing to the next port when --port is occupied.",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        default=None,
        help="Write the selected browser URL to this file after reserving the port.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--reload", action="store_true")
    add_runtime_startup_arguments(parser)
    return parser


def _socket_bind_host(host: str) -> str:
    value = str(host or DEFAULT_HOST).strip()
    if value in {"localhost", "::1"}:
        return "127.0.0.1" if value == "localhost" else value
    return value


def browser_host(host: str) -> str:
    value = str(host or DEFAULT_HOST).strip()
    if value in {"0.0.0.0", "::", "[::]"}:
        return DEFAULT_HOST
    return "127.0.0.1" if value == "localhost" else value


def reserve_available_socket(
    host: str,
    start_port: int,
    *,
    search_limit: int = DEFAULT_PORT_SEARCH_LIMIT,
    strict: bool = False,
) -> tuple[socket.socket, int]:
    """Reserve the first available TCP socket at or after ``start_port``.

    Returning the already-bound socket avoids a check-then-bind race where a
    second process could claim the selected port before Uvicorn starts.
    """

    requested_port = int(start_port)
    if requested_port < 1 or requested_port > 65535:
        raise ValueError("Port must be between 1 and 65535.")
    attempts = 1 if strict else max(1, int(search_limit))
    last_error: OSError | None = None
    bind_host = _socket_bind_host(host)

    for offset in range(attempts):
        port = requested_port + offset
        if port > 65535:
            break
        family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
        candidate = socket.socket(family, socket.SOCK_STREAM)
        try:
            candidate.bind((bind_host, port))
            candidate.listen(2048)
            candidate.set_inheritable(True)
            return candidate, port
        except OSError as exc:
            last_error = exc
            candidate.close()
            if strict:
                break

    tried_end = min(65535, requested_port + attempts - 1)
    detail = f" Could not bind ports {requested_port}-{tried_end} on {bind_host}."
    if last_error is not None:
        detail += f" Last error: {last_error}"
    raise OSError(detail.strip())


def selected_url(host: str, port: int) -> str:
    display_host = browser_host(host)
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{int(port)}"


def publish_selected_url(url: str, path: Path | None, *, console: TextIO | None = None) -> None:
    output = console
    if output is not None:
        print(f"IMAGE_GEN_WEBUI_URL: {url}", file=output, flush=True)
    if path is None:
        return
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(url + "\n", encoding="utf-8")
    temporary.replace(target)


def _load_saved_startup_settings(project_root: Path | None) -> dict:
    """Read WebUI startup settings before importing Torch/MSLK/xFormers.

    Invalid or incomplete project roots are handled later by the guarded startup
    path so the normal failure-bundle behavior remains intact.
    """

    try:
        context = ProjectContext.load(project_root=project_root)
    except Exception:
        return {}
    return WebUIStore(context.data_root / "webui").load_application_settings()


def _load_runtime():
    # Keep third-party imports inside the guarded startup path so missing WebUI
    # packages can still produce a local diagnostics bundle.
    import uvicorn
    from image_gen.webui.app import create_app

    return uvicorn, create_app


def _run_with_reserved_socket(args: argparse.Namespace) -> None:
    uvicorn, create_app = _load_runtime()
    reserved_socket, selected_port = reserve_available_socket(
        args.host,
        args.port,
        search_limit=args.port_search_limit,
        strict=args.strict_port,
    )
    url = selected_url(args.host, selected_port)
    publish_selected_url(url, args.url_file, console=None)
    if selected_port != args.port:
        print(
            f"Requested port {args.port} is occupied; using the next available port {selected_port}."
        )
    print(f"Starting server at {url}", flush=True)

    config = uvicorn.Config(
        create_app(
            project_root=args.project_root,
            runtime_startup_options=getattr(args, "runtime_startup_options", None),
        ),
        host=args.host,
        port=selected_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[reserved_socket])
    finally:
        reserved_socket.close()
        if args.url_file is not None:
            try:
                args.url_file.expanduser().resolve().unlink(missing_ok=True)
            except OSError:
                pass


def _main_impl(args: argparse.Namespace) -> None:
    if args.reload:
        uvicorn, _ = _load_runtime()
        # Reload mode needs to pass a listener into a child process. Resolve an
        # available port first, publish it, then let Uvicorn own the reload loop.
        reserved_socket, selected_port = reserve_available_socket(
            args.host,
            args.port,
            search_limit=args.port_search_limit,
            strict=args.strict_port,
        )
        reserved_socket.close()
        url = selected_url(args.host, selected_port)
        publish_selected_url(url, args.url_file, console=None)
        if selected_port != args.port:
            print(
                f"Requested port {args.port} is occupied; using the next available port {selected_port}."
            )
        print(f"Starting development server at {url}", flush=True)
        try:
            uvicorn.run(
                "image_gen.webui.server:development_app",
                host=args.host,
                port=selected_port,
                reload=True,
                factory=True,
            )
        finally:
            if args.url_file is not None:
                try:
                    args.url_file.expanduser().resolve().unlink(missing_ok=True)
                except OSError:
                    pass
        return
    _run_with_reserved_socket(args)


def main() -> None:
    raw_argv = list(sys.argv[1:])
    parser_argv = argv_for_primary_parser(raw_argv)
    args = build_parser().parse_args(parser_argv)
    args._runtime_argv = raw_argv
    bootstrap_runtime_startup(
        args,
        settings=_load_saved_startup_settings(args.project_root),
    )
    try:
        _main_impl(args)
    except Exception as exc:
        bundle = write_webui_failure_bundle(
            project_root=args.project_root,
            stage="startup",
            error=exc,
            extra={
                "host": args.host,
                "port": args.port,
                "reload": args.reload,
                "url_file": str(args.url_file) if args.url_file else None,
            },
        )
        print(f"IMAGE_GEN WebUI startup failed: {exc}", flush=True)
        print(f"Diagnostic bundle: {bundle}", flush=True)
        print(f"(failure bundle: {bundle})", flush=True)
        raise


def development_app():
    try:
        _, create_app = _load_runtime()
        return create_app()
    except Exception as exc:
        bundle = write_webui_failure_bundle(
            project_root=None,
            stage="development_startup",
            error=exc,
        )
        print(f"IMAGE_GEN development WebUI failed (failure bundle: {bundle})", flush=True)
        raise


if __name__ == "__main__":
    main()
