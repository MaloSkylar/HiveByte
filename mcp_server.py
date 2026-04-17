"""
FastMCP Server — self-hosted tool provider v4.4

    python mcp_server.py

File tools respect settings.json for access control:
  fs_access_enabled: true/false
  fs_allowed_paths: comma-separated list of directories
  fs_home_dir: primary file root for relative paths and writes

Relative file paths resolve from the configured file root.

SQL tools use SQLite databases in the allowed filesystem paths.
Trade journal tools persist to a SQLite DB at ./trade_journal.db (configurable).
"""

import ast
import base64
import csv
import hashlib
import inspect
import io
import ipaddress
import json
import math
import operator as _operator
import os
import random
import re
import secrets as _secrets
import shutil
import socket
import sqlite3
import string
import uuid
import datetime
import urllib.parse
import logging
from functools import wraps
from pathlib import Path
from typing import Annotated, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool
import uvicorn
from config import get_fs_access_scope, load_settings_snapshot
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from vector_store import (
    VectorStoreError,
    add_text as chroma_add_text,
    delete_collection as chroma_delete_collection,
    delete_document_ids as chroma_delete_document_ids,
    list_collections as chroma_list_collections,
    search as chroma_search,
)

log = logging.getLogger("local-mcp.server")

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"
TRADE_JOURNAL_DB = DATA_DIR / "trade_journal.db"

mcp = FastMCP(
    name="local_tools_mcp",
    instructions=(
        "Local MCP server exposing utility, text, file, web, email, SQL/database, "
        "trading-journal, system, and n8n workflow-automation tools."
    ),
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tool_field_annotation(field_info: Any) -> Any:
    """Build an Annotated type that preserves a model field's schema metadata."""
    metadata = [*field_info.metadata]
    if field_info.description is not None or field_info.json_schema_extra is not None:
        metadata.append(
            Field(
                description=field_info.description,
                json_schema_extra=field_info.json_schema_extra,
            )
        )
    return Annotated[field_info.annotation, *metadata] if metadata else field_info.annotation


def _flatten_model_tool(tool: FunctionTool) -> FunctionTool | None:
    """
    Flatten tools defined as `tool(params: BaseModel)` into top-level MCP args.

    FastMCP otherwise exposes these as a single nested `params` object, which
    makes them much harder for LLM agents to call correctly.
    """
    sig = inspect.signature(tool.fn)
    params = list(sig.parameters.values())
    if len(params) != 1:
        return None

    model_cls = params[0].annotation
    if not inspect.isclass(model_cls) or not issubclass(model_cls, BaseModel):
        return None

    flat_params = []
    annotations: dict[str, Any] = {
        "return": sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else str
    }
    for name, field_info in model_cls.model_fields.items():
        annotation = _tool_field_annotation(field_info)
        default = (
            inspect.Parameter.empty
            if field_info.is_required()
            else field_info.get_default(call_default_factory=True)
        )
        flat_params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
        annotations[name] = annotation

    flat_sig = inspect.Signature(flat_params, return_annotation=annotations["return"])

    @wraps(tool.fn)
    def wrapper(*args, **kwargs):
        bound = flat_sig.bind(*args, **kwargs)
        model = model_cls(**bound.arguments)
        return tool.fn(model)

    wrapper.__signature__ = flat_sig
    wrapper.__annotations__ = annotations

    description = tool.description.replace("params.", "") if tool.description else None
    return FunctionTool.from_function(
        wrapper,
        name=tool.name,
        version=tool.version,
        title=tool.title,
        description=description,
        icons=tool.icons,
        tags=tool.tags,
        annotations=tool.annotations,
        output_schema=tool.output_schema,
        meta=tool.meta,
        task=tool.task_config,
        serializer=tool.serializer,
        timeout=tool.timeout,
        auth=tool.auth,
    )


def _flatten_registered_model_tools() -> None:
    """Replace nested single-model tools with flat top-level argument schemas."""
    provider = mcp._local_provider
    for key, component in list(provider._components.items()):
        if not key.startswith("tool:") or not isinstance(component, FunctionTool):
            continue
        flattened = _flatten_model_tool(component)
        if flattened is None:
            continue
        provider.remove_tool(component.name, component.version)
        provider.add_tool(flattened)


def _err(msg: str, **extra) -> str:
    """Return a consistent JSON error payload."""
    return json.dumps({"error": msg, **extra})


def _http_headers() -> dict:
    return {"User-Agent": "Mozilla/5.0 (compatible; LocalMCP/4.4)"}


def _filesystem_roots(settings: dict) -> list[str]:
    """Return the configured file root plus any extra allowed paths."""
    scope = get_fs_access_scope(settings, fallback=WORKSPACE_DIR)
    return scope["allowed_paths"]


def _resolve_fs_path(requested_path: str, settings: dict | None = None) -> str:
    """
    Resolve filesystem tool paths.

    Relative paths are anchored to the configured file root so agents can use
    simple paths like './notes/todo.txt' without guessing the write location.
    """
    scope = get_fs_access_scope(settings, fallback=WORKSPACE_DIR)
    root = Path(scope["root"])
    req = Path(requested_path).expanduser()
    if req.is_absolute():
        return str(req.resolve())
    return str((root / req).resolve())


def _resolve_file_tool_path_argument(
    path: str = "",
    destination: str = "",
    source: str = "",
    file_path: str = "",
) -> tuple[bool, str, str]:
    """
    Accept common path-argument variations used by LLMs.

    Preferred name is `path`, but some models incorrectly send `destination`,
    `source`, or `file_path` for file-creation/write tools.
    """
    candidates = [
        ("path", path.strip()),
        ("destination", destination.strip()),
        ("file_path", file_path.strip()),
        ("source", source.strip()),
    ]
    provided = [(name, value) for name, value in candidates if value]
    if not provided:
        return False, "Provide 'path' for the target file.", ""
    unique_values = {value for _, value in provided}
    if len(unique_values) > 1:
        details = ", ".join(f"{name}='{value}'" for name, value in provided)
        return False, f"Conflicting target file arguments: {details}. Use one target path.", ""
    chosen_name, chosen_value = next((name, value) for name, value in candidates if value)
    return True, chosen_value, chosen_name


def _load_fs_settings() -> dict:
    snapshot = load_settings_snapshot()
    scope = get_fs_access_scope(snapshot, fallback=WORKSPACE_DIR)
    return {
        "fs_access_enabled": scope["enabled"],
        "fs_allowed_paths": scope["allowed_paths"],
        "fs_home_dir": scope["root"],
    }


def _path_under(child: str, parent: str) -> bool:
    """
    FIX [1] — safe directory-boundary check.
    Old naive startswith() let /home/user2 match /home/user.
    We normalise both paths and require the child to be exactly the parent
    or to start with parent + the OS path separator.
    """
    p = os.path.normcase(os.path.abspath(parent))
    c = os.path.normcase(os.path.abspath(child))
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


# ---------------------------------------------------------------------------
# SSRF guard (H-2) — block loopback / link-local / private / reserved targets
# unless the operator explicitly opts in with MCP_ALLOW_LOCAL_HTTP=1.
# ---------------------------------------------------------------------------

def _ssrf_allowed_local() -> bool:
    raw = os.getenv("MCP_ALLOW_LOCAL_HTTP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _ssrf_check_url(url: str) -> tuple[bool, str]:
    """Return (ok, reason). Reject non-http(s) schemes and private-network hosts
    unless MCP_ALLOW_LOCAL_HTTP=1. Used by http_request / web_fetch / etc."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return False, f"invalid URL: {exc}"
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return False, f"scheme '{scheme}' is not allowed; only http/https"
    host = parsed.hostname or ""
    if not host:
        return False, "URL has no host"
    if _ssrf_allowed_local():
        return True, ""
    # Resolve all addresses and reject if any is disallowed.
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as exc:
        return False, f"hostname resolution failed: {exc}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, (
                f"host '{host}' resolves to '{ip}' which is loopback/private/"
                "reserved; set MCP_ALLOW_LOCAL_HTTP=1 to allow"
            )
    return True, ""


# ---------------------------------------------------------------------------
# Safe math expression evaluator (H-3) — replaces eval() in calculator.
# Uses the AST to permit only numeric literals, +, -, *, /, //, %, **, unary
# +/-, function calls to an approved whitelist, and attribute-less name
# lookups into that whitelist. Rejects dunder access and any node type not
# in the allowlist below.
# ---------------------------------------------------------------------------

_SAFE_BIN_OPS = {
    ast.Add: _operator.add,
    ast.Sub: _operator.sub,
    ast.Mult: _operator.mul,
    ast.Div: _operator.truediv,
    ast.FloorDiv: _operator.floordiv,
    ast.Mod: _operator.mod,
    ast.Pow: _operator.pow,
}
_SAFE_UNARY_OPS = {ast.UAdd: _operator.pos, ast.USub: _operator.neg}


def _safe_math_eval(expression: str, allowed: dict[str, Any]) -> Any:
    if "_" in expression:
        raise ValueError("underscores are not allowed in expressions")
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.Num):  # py<3.8 fallback
            return node.n
        if isinstance(node, ast.BinOp):
            op = _SAFE_BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise ValueError(f"name '{node.id}' is not allowed")
            return allowed[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls are allowed")
            if node.func.id not in allowed:
                raise ValueError(f"function '{node.func.id}' is not allowed")
            func = allowed[node.func.id]
            if not callable(func):
                raise ValueError(f"'{node.func.id}' is not callable")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            args = [_eval(a) for a in node.args]
            return func(*args)
        if isinstance(node, ast.Tuple):
            return tuple(_eval(e) for e in node.elts)
        raise ValueError(f"syntax element {type(node).__name__} not allowed")

    return _eval(tree)


def _check_fs_access(requested_path: str, settings: dict | None = None) -> tuple[bool, str]:
    """Return (allowed, resolved_path_or_error_message)."""
    settings = settings or _load_fs_settings()
    if not settings["fs_access_enabled"]:
        return False, "File system access is disabled. Enable it in Settings -> File System Access."
    root_dir = settings["fs_home_dir"]
    allowed = _filesystem_roots(settings)
    resolved = _resolve_fs_path(requested_path, settings)
    for ap in allowed:
        if _path_under(resolved, ap):
            return True, resolved
    if len(allowed) <= 1:
        return False, (
            f"Path '{resolved}' is outside the configured file root ({root_dir}). "
            f"Relative paths are resolved from that directory."
        )
    return False, (
        f"Path '{resolved}' is not in allowed directories. "
        f"Allowed: {', '.join(allowed)}. Relative paths are resolved from {root_dir}."
    )


def _resolve_db_path(db_path: str) -> tuple[bool, str]:
    # SECURITY FIX [M-8]: Reject paths whose extension is not already a
    # SQLite one. Previously the function silently appended ".db", which
    # meant an LLM-supplied "./report.txt" became "./report.txt.db" and
    # was created as a database. Require a proper extension up-front so
    # callers cannot clobber or create files with surprising names.
    p = Path(db_path)
    if p.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
        return False, (
            f"Database path must end in .db, .sqlite, or .sqlite3 "
            f"(got '{p.suffix or '<none>'}')."
        )
    return _check_fs_access(db_path)


def _parse_json_object(raw: str, field_name: str) -> tuple[bool, dict[str, Any] | str]:
    if not raw.strip():
        return True, {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"{field_name} must be valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return False, f"{field_name} must be a JSON object."
    return True, parsed


def _parse_json_array(raw: str, field_name: str) -> tuple[bool, list[Any] | str]:
    if not raw.strip():
        return True, []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"{field_name} must be valid JSON: {exc}"
    if not isinstance(parsed, list):
        return False, f"{field_name} must be a JSON array."
    return True, parsed


def _n8n_settings() -> dict:
    snapshot = load_settings_snapshot()
    return {
        "n8n_enabled": bool(snapshot.get("n8n_enabled", False)),
        "n8n_url": str(snapshot.get("n8n_url", "http://localhost:5678")),
        "n8n_api_key": str(snapshot.get("n8n_api_key", "")),
    }


def _n8n_headers(s: dict) -> dict:
    h = {"Content-Type": "application/json"}
    if s["n8n_api_key"]:
        h["X-N8N-API-KEY"] = s["n8n_api_key"]
    return h


def _ensure_trade_journal() -> str:
    """Create the trade_journal table if it doesn't exist. Returns resolved path."""
    db = str(TRADE_JOURNAL_DB.resolve())
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            symbol      TEXT    NOT NULL,
            direction   TEXT    NOT NULL CHECK(direction IN ('long','short')),
            entry_price REAL    NOT NULL,
            exit_price  REAL,
            contracts   INTEGER NOT NULL DEFAULT 1,
            pnl         REAL,
            outcome     TEXT    CHECK(outcome IN ('win','loss','scratch','open')),
            session     TEXT,
            notes       TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db


# ===========================================================================
# SECTION 1 — CORE UTILITY TOOLS  (existing, revamped)
# ===========================================================================

class GetTimeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'UTC', 'US/Central', 'America/Chicago'")


@mcp.tool(
    name="get_current_time",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def get_current_time(params: GetTimeInput) -> str:
    """Return the current date, time, ISO string, and Unix timestamp for any IANA timezone.

    Args:
        params.timezone: IANA timezone name (default 'UTC').
    Returns:
        JSON with keys: datetime (ISO-8601), timezone, day_of_week, unix_timestamp.
    """
    # FIX [2]: use local variable instead of mutating Pydantic model field.
    tz_name = params.timezone
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now(datetime.timezone.utc)
        tz_name = "UTC (fallback – unrecognised timezone)"
    return json.dumps({
        "datetime": now.isoformat(),
        "timezone": tz_name,
        "day_of_week": now.strftime("%A"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "unix_timestamp": int(now.timestamp()),
    })


class CalculatorInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    expression: str = Field(..., description="Math expression, e.g. '2**10 + sqrt(144)'. Supports all math module functions.", min_length=1, max_length=500)


@mcp.tool(
    name="calculator",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def calculator(params: CalculatorInput) -> str:
    """Safely evaluate a mathematical expression using Python's math module.

    Supports: all math functions (sqrt, sin, cos, log, pow, etc.), abs, round.
    Does NOT allow arbitrary Python code — builtins are blocked.

    Args:
        params.expression: Math expression string.
    Returns:
        JSON with keys: expression, result (or error).
    """
    # SECURITY FIX [H-3]: The previous implementation used eval() with
    # {"__builtins__": {}}, which is not a sandbox — dunder traversal on
    # literals (e.g., `().__class__.__bases__[0].__subclasses__()`) can
    # still recover arbitrary objects. We now parse the expression with
    # ast.parse(mode="eval") and only permit numeric literals, arithmetic
    # operators, and calls into an explicit whitelist built from math.*.
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
    try:
        result = _safe_math_eval(params.expression, allowed)
        return json.dumps({"expression": params.expression, "result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc), "expression": params.expression})


class WordCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., description="Text to analyse", min_length=1)


@mcp.tool(
    name="word_count",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def word_count(params: WordCountInput) -> str:
    """Count words, characters, sentences, paragraphs, and estimate reading time.

    Args:
        params.text: Input text string.
    Returns:
        JSON with keys: characters, words, sentences, paragraphs, reading_time_seconds.
    """
    t = params.text
    words = t.split()
    sentences = [s for s in re.split(r'[.!?]+', t) if s.strip()]
    paragraphs = [p for p in t.split("\n\n") if p.strip()]
    reading_time = max(1, round(len(words) / 238))  # avg 238 wpm
    return json.dumps({
        "characters": len(t),
        "characters_no_spaces": len(t.replace(" ", "")),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": max(len(paragraphs), 1),
        "reading_time_seconds": reading_time * 60,
        "reading_time_minutes": reading_time,
    })


class UnitConverterInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    value: float = Field(..., description="Numeric value to convert")
    from_unit: str = Field(..., description="Source unit, e.g. 'km', 'celsius', 'kg', 'liters', 'mph', 'mb'")
    to_unit: str = Field(..., description="Target unit, e.g. 'miles', 'fahrenheit', 'lbs', 'gallons', 'kph', 'gb'")


@mcp.tool(
    name="unit_converter",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def unit_converter(params: UnitConverterInput) -> str:
    """Convert between units: distance, weight, temperature, volume, speed, and data size.

    Supported categories:
      Distance: km, miles, m, ft, cm, inches, yards, nautical_miles
      Weight:   kg, lbs, g, oz, tonnes, stones
      Volume:   liters, gallons, ml, fl_oz, cups, pints, quarts
      Speed:    mph, kph, m/s, knots
      Data:     bytes, kb, mb, gb, tb
      Temp:     celsius/c, fahrenheit/f, kelvin/k

    Args:
        params.value: Value to convert.
        params.from_unit / params.to_unit: Unit strings (case-insensitive).
    Returns:
        JSON with keys: input, output, category.
    """
    CONVERSIONS: dict[str, dict[str, float]] = {
        # Distance (base: meters)
        "m":             {"km": 0.001, "miles": 0.000621371, "ft": 3.28084, "cm": 100, "inches": 39.3701, "yards": 1.09361, "nautical_miles": 0.000539957},
        "km":            {"m": 1000, "miles": 0.621371, "ft": 3280.84, "cm": 100000, "inches": 39370.1, "yards": 1093.61, "nautical_miles": 0.539957},
        "miles":         {"km": 1.60934, "m": 1609.34, "ft": 5280, "cm": 160934, "inches": 63360, "yards": 1760, "nautical_miles": 0.868976},
        "ft":            {"m": 0.3048, "km": 0.0003048, "miles": 0.000189394, "cm": 30.48, "inches": 12, "yards": 0.333333},
        "cm":            {"m": 0.01, "km": 0.00001, "miles": 6.2137e-6, "ft": 0.0328084, "inches": 0.393701},
        "inches":        {"cm": 2.54, "m": 0.0254, "ft": 0.0833333, "yards": 0.0277778},
        "yards":         {"m": 0.9144, "ft": 3, "miles": 0.000568182},
        "nautical_miles":{"km": 1.852, "miles": 1.15078, "m": 1852},
        # Weight (base: kg)
        "kg":            {"lbs": 2.20462, "g": 1000, "oz": 35.274, "tonnes": 0.001, "stones": 0.157473},
        "lbs":           {"kg": 0.453592, "g": 453.592, "oz": 16, "stones": 0.0714286},
        "g":             {"kg": 0.001, "lbs": 0.00220462, "oz": 0.035274},
        "oz":            {"kg": 0.0283495, "lbs": 0.0625, "g": 28.3495},
        "tonnes":        {"kg": 1000, "lbs": 2204.62},
        "stones":        {"kg": 6.35029, "lbs": 14},
        # Volume (base: liters)
        "liters":        {"gallons": 0.264172, "ml": 1000, "fl_oz": 33.814, "cups": 4.22675, "pints": 2.11338, "quarts": 1.05669},
        "gallons":       {"liters": 3.78541, "ml": 3785.41, "fl_oz": 128, "cups": 16, "pints": 8, "quarts": 4},
        "ml":            {"liters": 0.001, "gallons": 0.000264172, "fl_oz": 0.033814, "cups": 0.00422675},
        "fl_oz":         {"liters": 0.0295735, "gallons": 0.0078125, "ml": 29.5735, "cups": 0.125},
        "cups":          {"liters": 0.236588, "gallons": 0.0625, "ml": 236.588, "fl_oz": 8},
        "pints":         {"liters": 0.473176, "gallons": 0.125},
        "quarts":        {"liters": 0.946353, "gallons": 0.25},
        # Speed
        "mph":           {"kph": 1.60934, "m/s": 0.44704, "knots": 0.868976},
        "kph":           {"mph": 0.621371, "m/s": 0.277778, "knots": 0.539957},
        "m/s":           {"mph": 2.23694, "kph": 3.6, "knots": 1.94384},
        "knots":         {"mph": 1.15078, "kph": 1.852, "m/s": 0.514444},
        # Data sizes (base: bytes)
        "bytes":         {"kb": 1/1024, "mb": 1/1048576, "gb": 1/1073741824, "tb": 1/1099511627776},
        "kb":            {"bytes": 1024, "mb": 1/1024, "gb": 1/1048576, "tb": 1/1073741824},
        "mb":            {"bytes": 1048576, "kb": 1024, "gb": 1/1024, "tb": 1/1048576},
        "gb":            {"bytes": 1073741824, "kb": 1048576, "mb": 1024, "tb": 1/1024},
        "tb":            {"bytes": 1099511627776, "kb": 1073741824, "mb": 1048576, "gb": 1024},
    }
    TEMP_ALIASES = {
        "celsius": "celsius", "c": "celsius",
        "fahrenheit": "fahrenheit", "f": "fahrenheit",
        "kelvin": "kelvin", "k": "kelvin",
    }
    fu, tu = params.from_unit.lower(), params.to_unit.lower()

    # Temperature conversions
    fn = TEMP_ALIASES.get(fu)
    tn = TEMP_ALIASES.get(tu)
    if fn and tn:
        v = params.value
        if fn == "celsius" and tn == "fahrenheit":
            result = v * 9 / 5 + 32
        elif fn == "fahrenheit" and tn == "celsius":
            result = (v - 32) * 5 / 9
        elif fn == "celsius" and tn == "kelvin":
            result = v + 273.15
        elif fn == "kelvin" and tn == "celsius":
            result = v - 273.15
        elif fn == "fahrenheit" and tn == "kelvin":
            result = (v - 32) * 5 / 9 + 273.15
        elif fn == "kelvin" and tn == "fahrenheit":
            result = (v - 273.15) * 9 / 5 + 32
        else:
            result = v
        return json.dumps({"input": {"value": params.value, "unit": params.from_unit},
                           "output": {"value": round(result, 6), "unit": params.to_unit},
                           "category": "temperature"})

    if fu in CONVERSIONS and tu in CONVERSIONS.get(fu, {}):
        result = params.value * CONVERSIONS[fu][tu]
        return json.dumps({"input": {"value": params.value, "unit": params.from_unit},
                           "output": {"value": round(result, 8), "unit": params.to_unit},
                           "category": "conversion"})

    supported = list(CONVERSIONS.keys()) + list(TEMP_ALIASES.keys())
    return _err(f"Unsupported conversion: '{params.from_unit}' → '{params.to_unit}'.",
                supported_units=sorted(set(supported)))


# ===========================================================================
# SECTION 2 — TEXT & DATA PROCESSING  (20 new tools — 9 here)
# ===========================================================================

class JsonFormatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    json_string: str = Field(..., description="Raw JSON string to process", min_length=1)
    action: str = Field(default="format", description="Action: 'format' (pretty-print), 'minify', or 'validate'")
    indent: int = Field(default=2, description="Indent spaces for 'format' action", ge=1, le=8)


@mcp.tool(
    name="json_format",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def json_format(params: JsonFormatInput) -> str:
    """Format, minify, or validate a JSON string.

    Args:
        params.json_string: Input JSON.
        params.action: 'format' | 'minify' | 'validate'.
        params.indent: Indent size for format action (default 2).
    Returns:
        JSON with keys: action, valid, result (or error).
    """
    try:
        parsed = json.loads(params.json_string)
    except json.JSONDecodeError as exc:
        return json.dumps({"valid": False, "error": str(exc), "action": params.action})

    if params.action == "minify":
        result = json.dumps(parsed, separators=(",", ":"))
    elif params.action == "validate":
        return json.dumps({"valid": True, "action": "validate",
                           "keys_at_root": list(parsed.keys()) if isinstance(parsed, dict) else None,
                           "type": type(parsed).__name__})
    else:
        result = json.dumps(parsed, indent=params.indent, ensure_ascii=False)

    return json.dumps({"action": params.action, "valid": True, "result": result,
                       "original_bytes": len(params.json_string), "result_bytes": len(result)})


class CsvToJsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    csv_string: str = Field(..., description="CSV text (with header row)", min_length=1)
    delimiter: str = Field(default=",", description="Column delimiter character, e.g. ',' or '\\t'", max_length=1)
    max_rows: int = Field(default=200, description="Maximum rows to include in output", ge=1, le=5000)


@mcp.tool(
    name="csv_to_json",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def csv_to_json(params: CsvToJsonInput) -> str:
    """Parse a CSV string (with header row) into a JSON array of objects.

    Args:
        params.csv_string: Raw CSV text.
        params.delimiter: Column separator (default ',').
        params.max_rows: Max rows to return (default 200).
    Returns:
        JSON with keys: columns, rows, row_count, truncated.
    """
    try:
        reader = csv.DictReader(io.StringIO(params.csv_string), delimiter=params.delimiter)
        rows = []
        for i, row in enumerate(reader):
            if i >= params.max_rows:
                break
            rows.append(dict(row))
        columns = list(rows[0].keys()) if rows else []
        return json.dumps({
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": len(rows) == params.max_rows,
        })
    except Exception as exc:
        return _err(str(exc))


class RegexMatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., description="Text to search", min_length=1)
    pattern: str = Field(..., description="Regular expression pattern", min_length=1)
    flags: str = Field(default="", description="Flag string: 'i' (ignore case), 'm' (multiline), 's' (dotall), any combo")
    max_matches: int = Field(default=50, description="Maximum number of matches to return", ge=1, le=500)


@mcp.tool(
    name="regex_match",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def regex_match(params: RegexMatchInput) -> str:
    """Test a regex pattern against text and return all matches with positions.

    Args:
        params.text: Input text to search.
        params.pattern: Regex pattern.
        params.flags: Optional flags: 'i', 'm', 's' (combinable, e.g. 'im').
        params.max_matches: Max results (default 50).
    Returns:
        JSON with keys: pattern, match_count, matches (list of {match, start, end, groups}).
    """
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
    compiled_flags = 0
    for ch in params.flags.lower():
        if ch in flag_map:
            compiled_flags |= flag_map[ch]
    try:
        matches = []
        for i, m in enumerate(re.finditer(params.pattern, params.text, compiled_flags)):
            if i >= params.max_matches:
                break
            matches.append({
                "match": m.group(0),
                "start": m.start(),
                "end": m.end(),
                "groups": list(m.groups()),
            })
        return json.dumps({
            "pattern": params.pattern,
            "flags": params.flags,
            "match_count": len(matches),
            "truncated": len(matches) == params.max_matches,
            "matches": matches,
        })
    except re.error as exc:
        return _err(f"Invalid regex pattern: {exc}")


class TextHashInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., description="Text to hash")
    algorithm: str = Field(default="sha256", description="Hash algorithm: 'md5', 'sha1', 'sha256', 'sha512'")
    encoding: str = Field(default="utf-8", description="Text encoding for hashing (default 'utf-8')")


@mcp.tool(
    name="text_hash",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def text_hash(params: TextHashInput) -> str:
    """Generate a cryptographic hash of a text string.

    Args:
        params.text: Input text.
        params.algorithm: 'md5', 'sha1', 'sha256' (default), or 'sha512'.
        params.encoding: Text encoding (default 'utf-8').
    Returns:
        JSON with keys: algorithm, hash, length_bytes.
    """
    alg = params.algorithm.lower()
    supported = {"md5", "sha1", "sha256", "sha512"}
    if alg not in supported:
        return _err(f"Unsupported algorithm '{alg}'. Choose from: {', '.join(supported)}.")
    try:
        h = hashlib.new(alg, params.text.encode(params.encoding))
        return json.dumps({"algorithm": alg, "hash": h.hexdigest(), "length_bytes": len(params.text.encode(params.encoding))})
    except Exception as exc:
        return _err(str(exc))


class Base64CodecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., description="Text to encode or decode", min_length=1)
    action: str = Field(default="encode", description="'encode' (text→base64) or 'decode' (base64→text)")
    encoding: str = Field(default="utf-8", description="Character encoding (default 'utf-8')")


@mcp.tool(
    name="base64_codec",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def base64_codec(params: Base64CodecInput) -> str:
    """Encode a string to base64 or decode a base64 string back to text.

    Args:
        params.text: Input string.
        params.action: 'encode' or 'decode'.
        params.encoding: Character encoding (default 'utf-8').
    Returns:
        JSON with keys: action, input_length, result.
    """
    try:
        if params.action == "encode":
            result = base64.b64encode(params.text.encode(params.encoding)).decode("ascii")
        elif params.action == "decode":
            result = base64.b64decode(params.text.encode("ascii")).decode(params.encoding)
        else:
            return _err("action must be 'encode' or 'decode'.")
        return json.dumps({"action": params.action, "input_length": len(params.text), "result": result, "result_length": len(result)})
    except Exception as exc:
        return _err(str(exc))


class TextDiffInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_a: str = Field(..., description="Original text")
    text_b: str = Field(..., description="New/modified text")
    context_lines: int = Field(default=3, description="Lines of context around each change", ge=0, le=20)


@mcp.tool(
    name="text_diff",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def text_diff(params: TextDiffInput) -> str:
    """Compare two text strings and return a unified diff with added/removed lines.

    Args:
        params.text_a: Original text.
        params.text_b: Modified text.
        params.context_lines: Context lines around changes (default 3).
    Returns:
        JSON with keys: additions, deletions, unchanged, diff_lines (list of {type, line}).
    """
    import difflib
    a_lines = params.text_a.splitlines(keepends=True)
    b_lines = params.text_b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a_lines, b_lines, n=params.context_lines, lineterm=""))
    result_lines = []
    additions = deletions = 0
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            result_lines.append({"type": "addition", "line": line[1:]})
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            result_lines.append({"type": "deletion", "line": line[1:]})
            deletions += 1
        elif line.startswith("@@"):
            result_lines.append({"type": "hunk", "line": line})
        else:
            result_lines.append({"type": "context", "line": line})
    return json.dumps({
        "additions": additions,
        "deletions": deletions,
        "identical": additions == 0 and deletions == 0,
        "diff_lines": result_lines,
    })


class GenerateUuidInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = Field(default=1, description="Number of UUIDs to generate", ge=1, le=100)
    # FIX [3]: restrict to 1 and 4 only. Old field (ge=1, le=4) let versions 2
    # and 3 through, which silently fell through to uuid4() — misleading.
    version: int = Field(
        default=4,
        description="UUID version: 4 (random, default) or 1 (time-based). Versions 2 and 3 are not supported.",
    )
    uppercase: bool = Field(default=False, description="Return UUIDs in uppercase")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v not in (1, 4):
            raise ValueError("Only UUID versions 1 and 4 are supported.")
        return v


@mcp.tool(
    name="generate_uuid",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def generate_uuid(params: GenerateUuidInput) -> str:
    """Generate one or more UUIDs (v1 time-based or v4 random).

    Args:
        params.count: Number to generate (1–100, default 1).
        params.version: 1 or 4 (default 4).
        params.uppercase: Return uppercase (default false).
    Returns:
        JSON with keys: version, count, uuids.
    """
    results = []
    for _ in range(params.count):
        u = uuid.uuid1() if params.version == 1 else uuid.uuid4()
        s = str(u).upper() if params.uppercase else str(u)
        results.append(s)
    return json.dumps({"version": params.version, "count": params.count,
                       "uuids": results, "first": results[0]})


class RandomDataInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    type: str = Field(default="integer", description="Type: 'integer', 'float', 'choice', 'password', 'hex_token'")
    min_val: float = Field(default=1, description="Minimum value (integer/float mode)")
    max_val: float = Field(default=100, description="Maximum value (integer/float mode)")
    choices: str = Field(default="", description="Comma-separated list for 'choice' type, e.g. 'heads,tails'")
    length: int = Field(default=16, description="Length for 'password' and 'hex_token' types", ge=4, le=256)
    count: int = Field(default=1, description="How many values to generate", ge=1, le=100)


@mcp.tool(
    name="random_data",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def random_data(params: RandomDataInput) -> str:
    """Generate random numbers, choices, passwords, or hex tokens.

    Types:
      'integer'   — random int between min_val and max_val
      'float'     — random float between min_val and max_val
      'choice'    — pick randomly from a comma-separated choices list
      'password'  — alphanumeric+symbol password of given length
      'hex_token' — cryptographically-suitable hex token of given length

    Args:
        params: RandomDataInput with type, range, choices, length, count fields.
    Returns:
        JSON with keys: type, values, count.
    """
    results = []
    # FIX [8]: validate type BEFORE the loop so we never return a partial
    # result and never evaluate unknown types on every iteration.
    t = params.type.lower()
    known_types = {"integer", "float", "choice", "password", "hex_token"}
    if t not in known_types:
        return _err(f"Unknown type '{params.type}'. Use: {', '.join(sorted(known_types))}.")

    if t == "choice":
        opts = [c.strip() for c in params.choices.split(",") if c.strip()]
        if not opts:
            return _err("Provide a comma-separated 'choices' list for type 'choice'.")
    else:
        opts = []

    for _ in range(params.count):
        if t == "integer":
            results.append(random.randint(int(params.min_val), int(params.max_val)))
        elif t == "float":
            results.append(round(random.uniform(params.min_val, params.max_val), 6))
        elif t == "choice":
            results.append(random.choice(opts))
        elif t == "password":
            # SECURITY FIX [L-3]: Use secrets.choice for password
            # generation; the previous random.choice is not suitable for
            # security-sensitive output.
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            results.append("".join(_secrets.choice(alphabet) for _ in range(params.length)))
        elif t == "hex_token":
            results.append(_secrets.token_hex(params.length // 2))
    return json.dumps({"type": params.type, "values": results,
                       "count": len(results), "first": results[0]})


class UrlEncodeDecodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., description="Text to encode or decode", min_length=1)
    action: str = Field(default="encode", description="'encode' (text→%XX) or 'decode' (%XX→text)")


@mcp.tool(
    name="url_encode_decode",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def url_encode_decode(params: UrlEncodeDecodeInput) -> str:
    """URL-encode or URL-decode a string (percent encoding).

    Args:
        params.text: Input string.
        params.action: 'encode' or 'decode'.
    Returns:
        JSON with keys: action, input, result.
    """
    if params.action == "encode":
        result = urllib.parse.quote(params.text, safe="")
    elif params.action == "decode":
        result = urllib.parse.unquote(params.text)
    else:
        return _err("action must be 'encode' or 'decode'.")
    return json.dumps({"action": params.action, "input": params.text, "result": result})


# ===========================================================================
# SECTION 3 — FILE SYSTEM TOOLS  (existing revamped + 3 new)
# ===========================================================================

class ListFilesInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    directory: str = Field(default=".", description="Use '.' for the configured file root, or relative paths like './projects'")
    max_entries: int = Field(default=50, description="Maximum entries to return", ge=1, le=500)
    show_hidden: bool = Field(default=False, description="Include hidden files/dirs (starting with '.')")
    extension_filter: str = Field(default="", description="Only show files with this extension, e.g. '.py' or '.csv'")


@mcp.tool(
    name="list_files",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def list_files(params: ListFilesInput) -> str:
    """List files in a directory with optional filtering by extension or hidden status.

    Args:
        params.directory: Target directory (default '.').
        params.max_entries: Max results (default 50).
        params.show_hidden: Include dotfiles (default false).
        params.extension_filter: Filter by extension, e.g. '.py'.
    Returns:
        JSON with keys: path, root, allowed_paths, entries (list of {name, type, size_bytes, modified}), shown, total.
    """
    settings = _load_fs_settings()
    root = settings["fs_home_dir"]
    ok, resolved = _check_fs_access(params.directory, settings)
    if not ok:
        return json.dumps({
            "error": resolved,
            "hint": f"Relative paths are resolved from the configured file root: {root}.",
        })
    try:
        entries = []
        total = 0
        for entry in sorted(os.scandir(resolved), key=lambda e: (not e.is_dir(), e.name.lower())):
            if not params.show_hidden and entry.name.startswith("."):
                continue
            if params.extension_filter and entry.is_file():
                if not entry.name.lower().endswith(params.extension_filter.lower()):
                    continue
            total += 1
            if len(entries) < params.max_entries:
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size_bytes": stat.st_size if entry.is_file() else None,
                    "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        result: dict[str, Any] = {
            "path": resolved,
            "root": root,
            "allowed_paths": settings["fs_allowed_paths"],
            "entries": entries,
            "shown": len(entries),
            "total": total,
        }
        if total > params.max_entries:
            result["note"] = f"Showing {params.max_entries} of {total}. Increase max_entries or use extension_filter."
        return json.dumps(result)
    except Exception as exc:
        return _err(str(exc))


class ReadFileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path: str = Field(..., description="Relative path from the configured file root, e.g. './projects/file.txt'")
    max_chars: int = Field(default=8000, description="Maximum characters to return", ge=1, le=100000)
    start_line: int = Field(default=0, description="Start reading from this line number (0 = beginning)", ge=0)
    end_line: int = Field(default=0, description="Stop at this line number (0 = no limit)", ge=0)


@mcp.tool(
    name="read_file",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def read_file(params: ReadFileInput) -> str:
    """Read a text file. Supports character limits and line range selection.

    Args:
        params.path: File path.
        params.max_chars: Character limit (default 8000).
        params.start_line: First line to return (0-indexed, default 0).
        params.end_line: Last line to return (0 = all, default 0).
    Returns:
        JSON with keys: path, content, truncated, size_bytes, total_lines.
    """
    ok, resolved = _check_fs_access(params.path)
    if not ok:
        return _err(resolved)
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            if params.start_line > 0 or params.end_line > 0:
                lines = fh.readlines()
                end = params.end_line if params.end_line > 0 else len(lines)
                selected = lines[params.start_line:end]
                raw = "".join(selected)
                content = raw[:params.max_chars]
                # FIX [6]: compare raw length vs limit using strict >.
                truncated = len(raw) > params.max_chars
                total_lines = len(lines)
            else:
                content = fh.read(params.max_chars)
                # Peek one more character to know whether the file continues.
                remainder = fh.read(1)
                truncated = len(remainder) > 0
                newline_count = content.count("\n") + remainder.count("\n")
                last_char = remainder or (content[-1] if content else "")
                for chunk in iter(lambda: fh.read(65536), ""):
                    if not chunk:
                        break
                    newline_count += chunk.count("\n")
                    last_char = chunk[-1]
                total_lines = 0 if not content and not remainder else newline_count + (0 if last_char == "\n" else 1)
        return json.dumps({
            "path": resolved,
            "content": content,
            "truncated": truncated,
            "size_bytes": os.path.getsize(resolved),
            "total_lines": total_lines,
        })
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="create_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def create_file(
    path: Annotated[str, "Target file path relative to the configured file root. Preferred argument name."] = "",
    content: Annotated[str, "Initial file content (defaults to empty string)"] = "",
    create_dirs: Annotated[bool, "Create parent directories if they do not exist (default true)"] = True,
    overwrite: Annotated[bool, "Allow replacing an existing file (default false)"] = False,
    destination: Annotated[str, "Compatibility alias for path. Leave empty unless a model sends this instead."] = "",
    source: Annotated[str, "Compatibility alias for path. Accepted for model compatibility when it is the same target path."] = "",
    file_path: Annotated[str, "Compatibility alias for path."] = "",
) -> str:
    """Create a new text file under the configured file root.

    Relative paths are resolved from the configured file root. By
    default, this tool refuses to replace existing files so agents have a
    straightforward, safe file-creation primitive.
    """
    settings = _load_fs_settings()
    path_ok, target_path, path_arg_name = _resolve_file_tool_path_argument(
        path=path,
        destination=destination,
        source=source,
        file_path=file_path,
    )
    if not path_ok:
        return _err(target_path)
    ok, resolved = _check_fs_access(target_path, settings)
    if not ok:
        return _err(resolved)
    resolved_path = Path(resolved)
    if resolved_path.exists() and resolved_path.is_dir():
        return _err(f"Path is a directory: {resolved}. This tool only creates files.")
    if resolved_path.exists() and not overwrite:
        return _err(f"File already exists: {resolved}. Use overwrite=true or call write_file to modify it.")
    try:
        if create_dirs:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved_path.exists()
        open_mode = "w" if overwrite else "x"
        with open(resolved_path, open_mode, encoding="utf-8") as fh:
            fh.write(content)
        return json.dumps({
            "path": resolved,
            "bytes_written": len(content.encode("utf-8")),
            "status": "overwritten" if existed else "created",
            "root": settings["fs_home_dir"],
            "path_argument": path_arg_name,
        })
    except FileExistsError:
        return _err(f"File already exists: {resolved}. Use overwrite=true or call write_file to modify it.")
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="write_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def write_file(
    path: Annotated[str, "Target file path relative to the configured file root. Preferred argument name."] = "",
    content: Annotated[str, "Text content to write"] = "",
    append: Annotated[bool, "Append to the end of the file instead of overwriting it (default false)"] = False,
    old_text: Annotated[str, "Optional exact text to find in the existing file. When provided, write_file edits the file in place."] = "",
    new_text: Annotated[str, "Replacement text for old_text. Can be empty to delete the matched text."] = "",
    replace_all: Annotated[bool, "Replace every occurrence of old_text instead of only the first one (default false)"] = False,
    create_dirs: Annotated[bool, "Create parent directories if they do not exist (default true)"] = True,
    destination: Annotated[str, "Compatibility alias for path. Leave empty unless a model sends this instead."] = "",
    source: Annotated[str, "Compatibility alias for path. Accepted for model compatibility when it is the same target path."] = "",
    file_path: Annotated[str, "Compatibility alias for path."] = "",
) -> str:
    """Write text to a file under the configured file root.

    Use this tool for both whole-file writes and small-model-friendly edits:
    overwrite by default, append when append=true, or replace old_text with
    new_text inside an existing file.
    """
    settings = _load_fs_settings()
    path_ok, target_path, path_arg_name = _resolve_file_tool_path_argument(
        path=path,
        destination=destination,
        source=source,
        file_path=file_path,
    )
    if not path_ok:
        return _err(target_path)
    edit_mode = bool(old_text)
    replacement_text = new_text
    if edit_mode and content:
        if new_text and content != new_text:
            return _err("Conflicting write_file arguments: content and new_text differ in edit mode.")
        replacement_text = content
    elif not edit_mode and new_text:
        if content and content != new_text:
            return _err("Conflicting write_file arguments: content and new_text differ. Use one whole-file payload.")
        content = new_text
    if append and edit_mode:
        return _err("Conflicting write_file arguments: append=true cannot be combined with old_text/new_text edit mode.")
    if replace_all and not edit_mode:
        return _err("Provide old_text when using replace_all=true.")
    ok, resolved = _check_fs_access(target_path, settings)
    if not ok:
        return _err(resolved)
    resolved_path = Path(resolved)
    if resolved_path.exists() and resolved_path.is_dir():
        return _err(f"Path is a directory: {resolved}. This tool only writes files.")
    try:
        if create_dirs:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved_path.exists()
        if edit_mode:
            if not existed:
                return _err(f"File does not exist: {resolved}. Create it first or omit old_text to do a normal write.")
            current = resolved_path.read_text(encoding="utf-8", errors="replace")
            matches_found = current.count(old_text)
            if matches_found == 0:
                return _err(f"Could not find old_text in file: {resolved}. Call read_file first and use an exact match.")
            if replace_all:
                updated = current.replace(old_text, replacement_text)
                replacements_applied = matches_found
            else:
                updated = current.replace(old_text, replacement_text, 1)
                replacements_applied = 1
            if updated == current:
                replacements_applied = 0
            resolved_path.write_text(updated, encoding="utf-8")
            return json.dumps({
                "path": resolved,
                "status": "edited" if replacements_applied else "unchanged",
                "operation": "replace",
                "replace_all": replace_all,
                "matches_found": matches_found,
                "replacements_applied": replacements_applied,
                "bytes_written": len(updated.encode("utf-8")),
                "root": settings["fs_home_dir"],
                "path_argument": path_arg_name,
            })
        write_mode = "a" if append else "w"
        with open(resolved_path, write_mode, encoding="utf-8") as fh:
            fh.write(content)
        status = "created" if not existed else ("appended" if append else "overwritten")
        return json.dumps({
            "path": resolved,
            "append": append,
            "operation": "append" if append else "write",
            "bytes_written": len(content.encode("utf-8")),
            "status": status,
            "root": settings["fs_home_dir"],
            "path_argument": path_arg_name,
        })
    except Exception as exc:
        return _err(str(exc))


# --- NEW: file_move, file_delete, directory_create ---

class FileMoveInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    source: str = Field(..., description="Source path relative to the configured file root, e.g. './old_name.txt'")
    destination: str = Field(..., description="Destination path relative to the configured file root, e.g. './archive/new_name.txt'")
    overwrite: bool = Field(default=False, description="Overwrite destination if it already exists (default false)")


@mcp.tool(
    name="file_move",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def file_move(params: FileMoveInput) -> str:
    """Move or rename a file within allowed directories.

    Args:
        params.source: Current file path.
        params.destination: New path (can be a rename or a move to another directory).
        params.overwrite: Allow overwriting destination (default false).
    Returns:
        JSON with keys: source, destination, status.
    """
    ok_src, src = _check_fs_access(params.source)
    if not ok_src:
        return _err(src)
    ok_dst, dst = _check_fs_access(params.destination)
    if not ok_dst:
        return _err(dst)
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return _err(f"Source file does not exist: {src}")
    if not src_path.is_file():
        return _err(f"Source path is a directory, not a file: {src}")
    if dst_path.exists() and dst_path.is_dir():
        return _err(f"Destination path is a directory, not a file: {dst}")
    if dst_path.exists() and not params.overwrite:
        return _err(f"Destination already exists: {dst}. Set overwrite=true to replace it.")
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and params.overwrite:
            dst_path.unlink()
        shutil.move(src, dst)
        return json.dumps({"source": src, "destination": dst, "status": "moved"})
    except Exception as exc:
        return _err(str(exc))


class FileDeleteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path: str = Field(..., description="Relative path of file to delete, e.g. './temp/old.txt'")
    confirm: bool = Field(default=False, description="Set to true to confirm deletion (safety guard)")


@mcp.tool(
    name="file_delete",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def file_delete(params: FileDeleteInput) -> str:
    """Delete a single file. Requires confirm=true as a safety guard.

    Args:
        params.path: File to delete.
        params.confirm: Must be true to proceed (default false).
    Returns:
        JSON with keys: path, status.
    """
    if not params.confirm:
        return _err("Set confirm=true to proceed with deletion. This action cannot be undone.")
    ok, resolved = _check_fs_access(params.path)
    if not ok:
        return _err(resolved)
    p = Path(resolved)
    if not p.exists():
        return _err(f"File does not exist: {resolved}")
    if p.is_dir():
        return _err(f"Path is a directory: {resolved}. This tool only deletes files.")
    try:
        size = p.stat().st_size
        p.unlink()
        return json.dumps({"path": resolved, "status": "deleted", "bytes_freed": size})
    except Exception as exc:
        return _err(str(exc))


class DirectoryCreateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path: str = Field(..., description="Relative path for the new directory, e.g. './data/reports/2026'")
    exist_ok: bool = Field(default=True, description="Do not error if directory already exists (default true)")


@mcp.tool(
    name="directory_create",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def directory_create(params: DirectoryCreateInput) -> str:
    """Create a directory (and any required parent directories).

    Args:
        params.path: Target directory path.
        params.exist_ok: Don't raise an error if already exists (default true).
    Returns:
        JSON with keys: path, status (created | already_exists).
    """
    ok, resolved = _check_fs_access(params.path)
    if not ok:
        return _err(resolved)
    p = Path(resolved)
    already = p.exists()
    try:
        p.mkdir(parents=True, exist_ok=params.exist_ok)
        return json.dumps({"path": resolved, "status": "already_exists" if already else "created"})
    except FileExistsError:
        return _err(f"Directory already exists: {resolved}. Set exist_ok=true to suppress this error.")
    except Exception as exc:
        return _err(str(exc))


# ===========================================================================
# SECTION 4 — SQL / DATABASE TOOLS  (existing revamped + 3 new)
# ===========================================================================

@mcp.tool(
    name="sql_create_database",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def sql_create_database(
    db_path: Annotated[str, "Relative path for new database, e.g. './mydata.db'"],
    tables_sql: Annotated[str, "SQL CREATE TABLE statements (semicolon-separated). Leave empty for blank database."] = "",
) -> str:
    """Create a new SQLite database, optionally initializing it with CREATE TABLE statements.

    Args:
        db_path: Relative path for the database file (.db extension added if omitted).
        tables_sql: Optional semicolon-separated CREATE TABLE SQL.
    Returns:
        JSON with keys: path, status (created|opened), tables_created.
    """
    ok, resolved = _resolve_db_path(db_path)
    if not ok:
        return _err(resolved)
    try:
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        existed = Path(resolved).exists()
        conn = sqlite3.connect(resolved)
        created_tables = []
        if tables_sql.strip():
            for stmt in tables_sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
                    m = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\']?(\w+)', stmt, re.IGNORECASE)
                    if m:
                        created_tables.append(m.group(1))
            conn.commit()
        conn.close()
        return json.dumps({"path": resolved, "status": "opened" if existed else "created", "tables_created": created_tables})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="sql_list_tables",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def sql_list_tables(
    db_path: Annotated[str, "Relative path to SQLite database, e.g. './mydata.db'"],
) -> str:
    """List all tables and views in a SQLite database.

    Args:
        db_path: Path to database file.
    Returns:
        JSON with keys: database, tables (list of {name, type}), count.
    """
    ok, resolved = _resolve_db_path(db_path)
    if not ok:
        return _err(resolved)
    if not Path(resolved).exists():
        return _err(f"Database not found: {resolved}")
    try:
        conn = sqlite3.connect(resolved)
        cursor = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
        tables = [{"name": r[0], "type": r[1]} for r in cursor.fetchall()]
        conn.close()
        return json.dumps({"database": resolved, "tables": tables, "count": len(tables)})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="sql_describe_table",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def sql_describe_table(
    db_path: Annotated[str, "Relative path to SQLite database"],
    table_name: Annotated[str, "Name of the table to describe"],
) -> str:
    """Show a table's schema, column types, primary keys, indexes, and row count.

    Args:
        db_path: Path to database file.
        table_name: Name of the table.
    Returns:
        JSON with keys: table, columns, row_count, indexes, foreign_keys.
    """
    ok, resolved = _resolve_db_path(db_path)
    if not ok:
        return _err(resolved)
    try:
        conn = sqlite3.connect(resolved)
        columns = [
            {"id": r[0], "name": r[1], "type": r[2], "not_null": bool(r[3]), "default": r[4], "primary_key": bool(r[5])}
            for r in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        ]
        if not columns:
            conn.close()
            return _err(f"Table '{table_name}' not found.")
        count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        indexes = [{"name": r[1], "unique": bool(r[2])} for r in conn.execute(f"PRAGMA index_list('{table_name}')").fetchall()]
        fk = [{"id": r[0], "from": r[3], "table": r[2], "to": r[4]} for r in conn.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()]
        conn.close()
        return json.dumps({"table": table_name, "columns": columns, "row_count": count, "indexes": indexes, "foreign_keys": fk})
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="sql_query",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def sql_query(
    db_path: Annotated[str, "Relative path to SQLite database"],
    query: Annotated[str, "SQL SELECT/PRAGMA/EXPLAIN/WITH query"],
    max_rows: Annotated[int, "Maximum rows to return (default 100)"] = 100,
) -> str:
    """Execute a read-only SQL query (SELECT/PRAGMA/EXPLAIN/WITH) and return JSON results.

    Args:
        db_path: Path to database file.
        query: SQL query string.
        max_rows: Row limit (default 100, max 5000).
    Returns:
        JSON with keys: columns, rows, row_count, truncated.
    """
    ok, resolved = _resolve_db_path(db_path)
    if not ok:
        return _err(resolved)
    stripped = query.strip().upper()
    if not any(stripped.startswith(k) for k in ("SELECT", "PRAGMA", "EXPLAIN", "WITH")):
        return _err("sql_query only allows SELECT/PRAGMA/EXPLAIN/WITH. Use sql_execute for writes.")
    try:
        conn = sqlite3.connect(resolved)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query)
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        rows = [dict(row) for i, row in enumerate(cursor) if i < max_rows]
        conn.close()
        return json.dumps({"columns": col_names, "rows": rows, "row_count": len(rows), "truncated": len(rows) >= max_rows})
    except Exception as exc:
        return _err(str(exc), query=query)


@mcp.tool(
    name="sql_execute",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def sql_execute(
    db_path: Annotated[str, "Relative path to SQLite database"],
    statement: Annotated[str, "SQL statement: INSERT, UPDATE, DELETE, CREATE, ALTER, DROP"],
) -> str:
    """Execute a write SQL statement and return rows affected. Use sql_query for SELECT.

    Args:
        db_path: Path to database file.
        statement: SQL write statement.
    Returns:
        JSON with keys: status, rows_affected, last_row_id.
    """
    ok, resolved = _resolve_db_path(db_path)
    if not ok:
        return _err(resolved)
    if statement.strip().upper().startswith("SELECT"):
        return _err("Use sql_query for SELECT statements.")
    try:
        conn = sqlite3.connect(resolved)
        cursor = conn.execute(statement)
        conn.commit()
        result = {"status": "ok", "rows_affected": cursor.rowcount, "last_row_id": cursor.lastrowid}
        conn.close()
        return json.dumps(result)
    except Exception as exc:
        return _err(str(exc), statement=statement)


# --- NEW: sql_export_csv, sql_import_csv, sql_backup ---

class SqlExportCsvInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    db_path: str = Field(..., description="Relative path to SQLite database")
    query: str = Field(..., description="SELECT query whose results will be exported")
    output_path: str = Field(..., description="Relative path for the output .csv file, e.g. './exports/data.csv'")
    include_header: bool = Field(default=True, description="Include column header row (default true)")


@mcp.tool(
    name="sql_export_csv",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def sql_export_csv(params: SqlExportCsvInput) -> str:
    """Export a SQL query result to a CSV file.

    Args:
        params.db_path: Source database path.
        params.query: SELECT statement to export.
        params.output_path: Destination .csv file path.
        params.include_header: Write column names as first row (default true).
    Returns:
        JSON with keys: output_path, rows_exported, columns.
    """
    ok_db, resolved_db = _resolve_db_path(params.db_path)
    if not ok_db:
        return _err(resolved_db)
    ok_out, resolved_out = _check_fs_access(params.output_path)
    if not ok_out:
        return _err(resolved_out)
    if not params.query.strip().upper().startswith("SELECT"):
        return _err("Only SELECT queries are supported for export.")
    try:
        conn = sqlite3.connect(resolved_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(params.query)
        rows = cursor.fetchall()
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()
        Path(resolved_out).parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if params.include_header:
                writer.writerow(col_names)
            writer.writerows([list(r) for r in rows])
        return json.dumps({"output_path": resolved_out, "rows_exported": len(rows), "columns": col_names})
    except Exception as exc:
        return _err(str(exc))


class SqlImportCsvInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    db_path: str = Field(..., description="Relative path to SQLite database (created if missing)")
    csv_path: str = Field(..., description="Relative path to the .csv file to import")
    table_name: str = Field(..., description="Target table name (created automatically if it doesn't exist)")
    has_header: bool = Field(default=True, description="CSV has a header row (default true)")
    if_exists: str = Field(default="append", description="'append' to add rows, or 'replace' to drop and recreate the table")


@mcp.tool(
    name="sql_import_csv",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def sql_import_csv(params: SqlImportCsvInput) -> str:
    """Import a CSV file into a SQLite table (auto-creates the table from headers).

    Args:
        params.db_path: Target database.
        params.csv_path: Source CSV file.
        params.table_name: Table to import into.
        params.has_header: CSV has headers (default true).
        params.if_exists: 'append' or 'replace' (default 'append').
    Returns:
        JSON with keys: table, rows_imported, columns, status.
    """
    ok_db, resolved_db = _resolve_db_path(params.db_path)
    if not ok_db:
        return _err(resolved_db)
    ok_csv, resolved_csv = _check_fs_access(params.csv_path)
    if not ok_csv:
        return _err(resolved_csv)
    if not Path(resolved_csv).exists():
        return _err(f"CSV file not found: {resolved_csv}")
    try:
        with open(resolved_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        if not all_rows:
            return _err("CSV file is empty.")
        if params.has_header:
            headers = [h.strip().replace(" ", "_").replace("-", "_") for h in all_rows[0]]
            data_rows = all_rows[1:]
        else:
            headers = [f"col{i}" for i in range(len(all_rows[0]))]
            data_rows = all_rows

        conn = sqlite3.connect(resolved_db)
        if params.if_exists == "replace":
            conn.execute(f'DROP TABLE IF EXISTS "{params.table_name}"')
        col_defs = ", ".join(f'"{h}" TEXT' for h in headers)
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{params.table_name}" ({col_defs})')
        placeholders = ", ".join("?" for _ in headers)
        conn.executemany(f'INSERT INTO "{params.table_name}" VALUES ({placeholders})', data_rows)
        conn.commit()
        conn.close()
        return json.dumps({"table": params.table_name, "rows_imported": len(data_rows), "columns": headers, "status": "ok"})
    except Exception as exc:
        return _err(str(exc))


class SqlBackupInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    db_path: str = Field(..., description="Relative path to the source SQLite database")
    backup_path: str = Field(default="", description="Destination path. Leave empty to auto-generate a timestamped backup alongside the source.")


@mcp.tool(
    name="sql_backup",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def sql_backup(params: SqlBackupInput) -> str:
    """Create a hot backup copy of a SQLite database.

    Args:
        params.db_path: Source database.
        params.backup_path: Destination (auto-generated timestamp path if empty).
    Returns:
        JSON with keys: source, backup, size_bytes, status.
    """
    ok_src, resolved_src = _resolve_db_path(params.db_path)
    if not ok_src:
        return _err(resolved_src)
    if not Path(resolved_src).exists():
        return _err(f"Source database not found: {resolved_src}")

    if not params.backup_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(resolved_src).stem
        backup_path_str = str(Path(resolved_src).parent / f"{stem}_backup_{ts}.db")
    else:
        ok_dst, backup_path_str = _check_fs_access(params.backup_path)
        if not ok_dst:
            return _err(backup_path_str)

    # FIX [7]: always close both connections via finally blocks.
    src_conn = sqlite3.connect(resolved_src)
    dst_conn = sqlite3.connect(backup_path_str)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    try:
        size = Path(backup_path_str).stat().st_size
        return json.dumps({"source": resolved_src, "backup": backup_path_str, "size_bytes": size, "status": "ok"})
    except Exception as exc:
        return _err(str(exc))


# ===========================================================================
# SECTION 5 — WEB TOOLS  (existing, revamped)
# ===========================================================================

@mcp.tool(
    name="web_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def web_search(
    query: Annotated[str, "Search query text"],
    max_results: Annotated[int, "Number of results to return (default 5, max 20)"] = 5,
) -> str:
    """Search the web using DuckDuckGo and return title, URL, and snippet for each result.

    Args:
        query: Search query.
        max_results: Number of results (1–20, default 5).
    Returns:
        JSON with keys: query, results (list of {title, url, snippet}), count.
    """
    try:
        import httpx
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=_http_headers(),
            timeout=10, follow_redirects=True,
        )
        resp.raise_for_status()
        snippets = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
            resp.text, re.DOTALL,
        )
        results = []
        for url, title, snippet in snippets[:max_results]:
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            if title:
                results.append({"title": title, "url": url, "snippet": snippet})
        return json.dumps({"query": query, "results": results, "count": len(results)})
    except Exception as exc:
        return _err(str(exc), query=query)


@mcp.tool(
    name="web_fetch",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def web_fetch(
    url: Annotated[str, "URL to fetch"],
    max_chars: Annotated[int, "Maximum characters to return (default 5000)"] = 5000,
) -> str:
    """Fetch a URL and return raw response body (HTML/text/JSON).

    Args:
        url: Full URL including scheme.
        max_chars: Character limit on returned body (default 5000).
    Returns:
        JSON with keys: url, status, content_type, content, truncated.
    """
    try:
        # SECURITY FIX [H-2]: SSRF guard.
        ok, reason = _ssrf_check_url(url)
        if not ok:
            return _err(f"blocked by SSRF guard: {reason}", url=url)
        import httpx
        resp = httpx.get(url, headers=_http_headers(), timeout=15, follow_redirects=True)
        if str(resp.url) != url:
            ok2, reason2 = _ssrf_check_url(str(resp.url))
            if not ok2:
                return _err(f"redirect blocked by SSRF guard: {reason2}", url=str(resp.url))
        resp.raise_for_status()
        content = resp.text[:max_chars]
        return json.dumps({
            "url": url, "status": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "content": content, "truncated": len(resp.text) > max_chars,
        })
    except Exception as exc:
        return _err(str(exc), url=url)


@mcp.tool(
    name="extract_text",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def extract_text(
    url: Annotated[str, "URL to extract clean text from"],
    max_chars: Annotated[int, "Maximum characters (default 4000)"] = 4000,
) -> str:
    """Fetch a web page, strip all HTML/JS/CSS, and return clean readable text.

    Args:
        url: Full URL.
        max_chars: Character limit (default 4000).
    Returns:
        JSON with keys: url, text, truncated, word_count.
    """
    try:
        # SECURITY FIX [H-2]: SSRF guard.
        ok, reason = _ssrf_check_url(url)
        if not ok:
            return _err(f"blocked by SSRF guard: {reason}", url=url)
        import httpx
        resp = httpx.get(url, headers=_http_headers(), timeout=15, follow_redirects=True)
        if str(resp.url) != url:
            ok2, reason2 = _ssrf_check_url(str(resp.url))
            if not ok2:
                return _err(f"redirect blocked by SSRF guard: {reason2}", url=str(resp.url))
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r'<(script|style|noscript|head)[^>]*>[\s\S]*?</\1>', '', html, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:max_chars]
        return json.dumps({"url": url, "text": text, "truncated": len(text) == max_chars,
                           "word_count": len(text.split())})
    except Exception as exc:
        return _err(str(exc), url=url)


_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


class HttpRequestInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    url: str = Field(..., description="Full URL including scheme, e.g. 'https://api.example.com/v1/items'")
    method: str = Field(default="GET", description="HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS")
    body: str = Field(default="", description="Request body for POST/PUT/PATCH (raw string or JSON)")
    headers_json: str = Field(default="{}", description="Extra headers as a JSON object, e.g. '{\"Authorization\": \"Bearer TOKEN\"}'")
    timeout: int = Field(default=15, description="Request timeout in seconds", ge=1, le=60)

    # FIX [9]: validate method before it reaches httpx to avoid cryptic errors.
    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _HTTP_METHODS:
            raise ValueError(
                f"Unknown HTTP method '{v}'. Supported: {', '.join(sorted(_HTTP_METHODS))}."
            )
        return upper


@mcp.tool(
    name="http_request",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def http_request(params: HttpRequestInput) -> str:
    """Make an HTTP request (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS) — useful for REST APIs.

    Args:
        params.url: Target URL.
        params.method: HTTP method, pre-validated (default GET).
        params.body: Request body for POST/PUT/PATCH.
        params.headers_json: Extra headers as JSON string.
        params.timeout: Timeout seconds (default 15).
    Returns:
        JSON with keys: status, headers, body, truncated.
    """
    try:
        # SECURITY FIX [H-2]: Reject loopback / private / link-local / reserved
        # targets unless MCP_ALLOW_LOCAL_HTTP=1. Blocks SSRF pivots into
        # 169.254.169.254 cloud metadata, 127.0.0.1 services, LAN hosts, etc.
        ok, reason = _ssrf_check_url(params.url)
        if not ok:
            return _err(f"blocked by SSRF guard: {reason}", url=params.url)
        import httpx
        extra_headers = json.loads(params.headers_json) if params.headers_json else {}
        hdrs = {**_http_headers(), **extra_headers}
        # params.method is already uppercased and validated by the Pydantic validator.
        with httpx.Client(timeout=params.timeout, follow_redirects=False) as client:
            if params.method in ("POST", "PUT", "PATCH"):
                resp = client.request(params.method, params.url, content=params.body, headers=hdrs)
            else:
                resp = client.request(params.method, params.url, headers=hdrs)
        # Manually follow redirects with an SSRF check on each hop.
        hops = 0
        while resp.is_redirect and hops < 5:
            next_url = str(resp.headers.get("location", ""))
            if not next_url:
                break
            next_url = str(resp.url.join(next_url))
            ok, reason = _ssrf_check_url(next_url)
            if not ok:
                return _err(
                    f"redirect blocked by SSRF guard: {reason}", url=next_url
                )
            hops += 1
            with httpx.Client(timeout=params.timeout, follow_redirects=False) as client:
                resp = client.request("GET", next_url, headers=hdrs)
        return json.dumps({
            "status": resp.status_code,
            "headers": dict(resp.headers.items()),
            "body": resp.text[:3000],
            "truncated": len(resp.text) > 3000,
        })
    except Exception as exc:
        return _err(str(exc), url=params.url)


@mcp.tool(
    name="url_info",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def url_info(url: Annotated[str, "URL to inspect"]) -> str:
    """Inspect a URL — returns status, headers, content type, redirect chain, and server info.

    Args:
        url: Full URL.
    Returns:
        JSON with keys: url, final_url, status, content_type, content_length, server, redirected.
    """
    try:
        # SECURITY FIX [H-2]: SSRF guard.
        ok, reason = _ssrf_check_url(url)
        if not ok:
            return _err(f"blocked by SSRF guard: {reason}", url=url)
        import httpx
        resp = httpx.get(url, headers=_http_headers(), timeout=10, follow_redirects=True)
        final_url = str(resp.url)
        if final_url != url:
            ok2, reason2 = _ssrf_check_url(final_url)
            if not ok2:
                return _err(f"redirect blocked by SSRF guard: {reason2}", url=final_url)
        return json.dumps({
            "url": url, "final_url": final_url,
            "status": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "content_length": resp.headers.get("content-length", "unknown"),
            "server": resp.headers.get("server", "unknown"),
            "redirected": final_url != url,
        })
    except Exception as exc:
        return _err(str(exc), url=url)


@mcp.tool(
    name="web_news",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def web_news(
    query: Annotated[str, "News search query"],
    max_results: Annotated[int, "Number of results (default 5)"] = 5,
) -> str:
    """Search for recent news headlines via DuckDuckGo News.

    Args:
        query: News topic or keyword query.
        max_results: Results to return (1–20, default 5).
    Returns:
        JSON with keys: query, results (list of {title, url, snippet}), count.
    """
    try:
        import httpx
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": f"{query} news", "df": "w"},
            headers=_http_headers(),
            timeout=10, follow_redirects=True,
        )
        resp.raise_for_status()
        snippets = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
            resp.text, re.DOTALL,
        )
        results = []
        for url, title, snippet in snippets[:max_results]:
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            if title:
                results.append({"title": title, "url": url, "snippet": snippet})
        return json.dumps({"query": query, "results": results, "count": len(results)})
    except Exception as exc:
        return _err(str(exc), query=query)


# ===========================================================================
# SECTION 6 — EMAIL
# ===========================================================================

class SendEmailInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    to: str = Field(..., description="Recipient email address(es), comma-separated for multiple")
    subject: str = Field(..., description="Email subject line", min_length=1, max_length=998)
    body: str = Field(..., description="Email body text (plain text or HTML depending on html_mode)")
    cc: str = Field(default="", description="CC recipients (comma-separated, optional)")
    bcc: str = Field(default="", description="BCC recipients (comma-separated, optional)")
    html_mode: bool = Field(default=False, description="Send body as HTML email (default: plain text)")


@mcp.tool(
    name="send_email",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def send_email(params: SendEmailInput) -> str:
    """Send an email via SMTP. Supports plain text and HTML, CC/BCC.

    Requires env vars: SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS, SMTP_FROM.

    Args:
        params: SendEmailInput with to, subject, body, cc, bcc, html_mode fields.
    Returns:
        JSON with keys: status, to, subject, recipients_total.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    smtp_settings = load_settings_snapshot()
    host = str(smtp_settings.get("smtp_host") or os.getenv("SMTP_HOST", ""))
    port = int(smtp_settings.get("smtp_port") or os.getenv("SMTP_PORT", "587"))
    user = str(smtp_settings.get("smtp_user") or os.getenv("SMTP_USER", ""))
    passwd = str(smtp_settings.get("smtp_pass") or os.getenv("SMTP_PASS", ""))
    sender = str(smtp_settings.get("smtp_from") or os.getenv("SMTP_FROM", user) or user)

    if not all([host, user, passwd]):
        return _err("Email not configured. Set SMTP settings in data/settings.json or the SMTP_* environment variables.")

    try:
        if params.html_mode:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(params.body, "html"))
        else:
            msg = MIMEText(params.body, "plain")

        msg["Subject"] = params.subject
        msg["From"] = sender
        msg["To"] = params.to
        if params.cc:
            msg["Cc"] = params.cc
        recipients = [e.strip() for e in params.to.split(",") if e.strip()]
        if params.cc:
            recipients += [e.strip() for e in params.cc.split(",") if e.strip()]
        if params.bcc:
            recipients += [e.strip() for e in params.bcc.split(",") if e.strip()]

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, passwd)
            server.send_message(msg, to_addrs=recipients)
        return json.dumps({"status": "sent", "to": params.to, "subject": params.subject,
                           "recipients_total": len(recipients), "html": params.html_mode})
    except Exception as exc:
        return _err(str(exc))


# ===========================================================================
# SECTION 7 — TRADING TOOLS  (revamped stubs + 3 new)
# ===========================================================================

class MarketQuoteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Ticker symbol, e.g. 'AAPL', 'MNQM5', 'BTC-USD'", min_length=1, max_length=20)
    source: str = Field(default="yahoo", description="Data source: 'yahoo' (yfinance) or 'alphavantage'")


@mcp.tool(
    name="market_quote",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def market_quote(params: MarketQuoteInput) -> str:
    """Fetch a live or delayed market quote for a symbol via yfinance or Alpha Vantage.

    Sources:
      'yahoo'        — uses yfinance (no API key needed, free)
      'alphavantage' — uses Alpha Vantage REST API (requires ALPHAVANTAGE_API_KEY env var)

    Args:
        params.symbol: Ticker symbol.
        params.source: 'yahoo' (default) or 'alphavantage'.
    Returns:
        JSON with price, volume, open, high, low, previous_close, change_pct, and market_time.
    """
    sym = params.symbol.upper()

    if params.source == "yahoo":
        try:
            import yfinance as yf  # type: ignore
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            return json.dumps({
                "symbol": sym,
                "source": "yahoo",
                "price": info.last_price,
                "previous_close": info.previous_close,
                "open": info.open,
                "day_high": info.day_high,
                "day_low": info.day_low,
                "volume": info.last_volume,
                "change": round(info.last_price - info.previous_close, 4) if info.previous_close else None,
                "change_pct": round((info.last_price - info.previous_close) / info.previous_close * 100, 3) if info.previous_close else None,
                "market_time": datetime.datetime.now().isoformat(),
            })
        except ImportError:
            return _err("yfinance not installed. Run: pip install yfinance", hint="Or use source='alphavantage' with ALPHAVANTAGE_API_KEY env var.")
        except Exception as exc:
            return _err(str(exc), symbol=sym)

    elif params.source == "alphavantage":
        provider_settings = load_settings_snapshot()
        api_key = str(provider_settings.get("alphavantage_api_key") or os.getenv("ALPHAVANTAGE_API_KEY", ""))
        if not api_key:
            return _err("Set Alpha Vantage credentials in data/settings.json or ALPHAVANTAGE_API_KEY.", hint="Free key at: https://www.alphavantage.co/support/#api-key")
        try:
            import httpx
            resp = httpx.get(
                "https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": sym, "apikey": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("Global Quote", {})
            if not data:
                return _err(f"No data returned for symbol '{sym}'.")
            return json.dumps({
                "symbol": sym,
                "source": "alphavantage",
                "price": float(data.get("05. price", 0)),
                "open": float(data.get("02. open", 0)),
                "day_high": float(data.get("03. high", 0)),
                "day_low": float(data.get("04. low", 0)),
                "volume": int(data.get("06. volume", 0)),
                "previous_close": float(data.get("08. previous close", 0)),
                "change": float(data.get("09. change", 0)),
                "change_pct": data.get("10. change percent", "0%"),
                "latest_trading_day": data.get("07. latest trading day"),
            })
        except Exception as exc:
            return _err(str(exc), symbol=sym)
    else:
        return _err(f"Unknown source '{params.source}'. Use 'yahoo' or 'alphavantage'.")


class TopstepxStatusInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    account_id: str = Field(default="", description="TopstepX account ID (optional)")


@mcp.tool(
    name="topstepx_status",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
def topstepx_status(params: TopstepxStatusInput) -> str:
    """Check TopstepX combine account status via the TopstepX API.

    Requires TOPSTEPX_API_KEY environment variable.

    Args:
        params.account_id: TopstepX account ID.
    Returns:
        JSON with account status, balance, drawdown info, and rules state.
    """
    provider_settings = load_settings_snapshot()
    api_key = str(provider_settings.get("topstepx_api_key") or os.getenv("TOPSTEPX_API_KEY", ""))
    if not api_key:
        return json.dumps({
            "status": "not_configured",
            "note": "Set TopstepX credentials in data/settings.json or TOPSTEPX_API_KEY to enable live account status.",
            "account_id": params.account_id or "not provided",
        })
    try:
        import httpx
        endpoint = "https://api.topstepx.com/api/v1/account"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        params_dict = {"accountId": params.account_id} if params.account_id else {}
        resp = httpx.get(endpoint, headers=headers, params=params_dict, timeout=10)
        resp.raise_for_status()
        return json.dumps({"status": "ok", "data": resp.json()})
    except Exception as exc:
        return _err(str(exc), hint="Verify TOPSTEPX_API_KEY and account_id are correct.")


# --- NEW: trade_journal_add, trade_journal_stats ---

class TradeJournalAddInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(..., description="Instrument symbol, e.g. 'MNQM5'", min_length=1, max_length=20)
    direction: str = Field(..., description="Trade direction: 'long' or 'short'")
    entry_price: float = Field(..., description="Entry price", gt=0)
    exit_price: float = Field(default=0.0, description="Exit price (0 if trade still open)", ge=0)
    contracts: int = Field(default=1, description="Number of contracts/shares/units", ge=1, le=1000)
    # FIX [4]: use Optional[float] so that a real P&L of 0.0 (scratch trade)
    # is stored as 0.0, not NULL. None means "not yet known".
    pnl: float | None = Field(
        default=None,
        description="Realized P&L (positive=profit, negative=loss). Omit if unknown; use 0.0 for scratch trades.",
    )
    outcome: str = Field(default="open", description="Trade result: 'win', 'loss', 'scratch', or 'open'")
    session: str = Field(default="", description="Session label, e.g. 'pre_market', 'combine_day_3'")
    notes: str = Field(default="", description="Free-text notes, e.g. 'revenge trade, ignored daily loss limit'")

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")
        return v

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("win", "loss", "scratch", "open"):
            raise ValueError("outcome must be 'win', 'loss', 'scratch', or 'open'")
        return v


@mcp.tool(
    name="trade_journal_add",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def trade_journal_add(params: TradeJournalAddInput) -> str:
    """Log a trade entry to the local trade journal database.

    Persists to trade_journal.db alongside this server file. Creates the table if needed.

    Args:
        params: TradeJournalAddInput — symbol, direction, entry/exit price, contracts, PnL, outcome, session, notes.
    Returns:
        JSON with keys: id, symbol, direction, pnl, outcome, logged_at.
    """
    try:
        db = _ensure_trade_journal()
        exit_val = params.exit_price if params.exit_price > 0 else None
        # pnl is already Optional[float]; None = unknown, 0.0 = scratch trade P&L.
        conn = sqlite3.connect(db)
        cursor = conn.execute(
            """INSERT INTO trades (symbol, direction, entry_price, exit_price, contracts, pnl, outcome, session, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (params.symbol.upper(), params.direction, params.entry_price, exit_val,
             params.contracts, params.pnl, params.outcome, params.session, params.notes),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logged = conn.execute("SELECT logged_at FROM trades WHERE id=?", (row_id,)).fetchone()[0]
        conn.close()
        return json.dumps({
            "id": row_id,
            "symbol": params.symbol.upper(),
            "direction": params.direction,
            "entry_price": params.entry_price,
            "exit_price": exit_val,
            "contracts": params.contracts,
            "pnl": params.pnl,
            "outcome": params.outcome,
            "session": params.session,
            "logged_at": logged,
        })
    except Exception as exc:
        return _err(str(exc))


class TradeJournalStatsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    symbol: str = Field(default="", description="Filter by symbol (empty = all symbols)")
    since_date: str = Field(default="", description="Filter trades on or after this date, format 'YYYY-MM-DD'")
    session: str = Field(default="", description="Filter by session label (empty = all sessions)")
    recent_limit: int = Field(
        default=20,
        description="Max trades to include in the recent_trades preview list (default 20). Does NOT affect statistics.",
        ge=1,
        le=500,
    )


@mcp.tool(
    name="trade_journal_stats",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def trade_journal_stats(params: TradeJournalStatsInput) -> str:
    """Query trade journal summary statistics: win rate, avg P&L, profit factor, streaks, recent trades.

    FIX [5]: aggregate statistics (win_rate, avg_pnl, profit_factor, etc.) are
    now computed over ALL matching closed trades. The recent_limit parameter
    only caps the recent_trades preview list — it never affects the numbers.

    Args:
        params.symbol: Filter by symbol (optional).
        params.since_date: ISO date filter (optional).
        params.session: Filter by session label (optional).
        params.recent_limit: Max entries in recent_trades preview (default 20).
    Returns:
        JSON — total_trades, wins, losses, scratch, win_rate, avg_pnl, avg_win,
               avg_loss, profit_factor, total_pnl, max_win, max_loss,
               current_streak, filters, recent_trades.
    """
    try:
        db = _ensure_trade_journal()
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row

        conditions = ["outcome != 'open'"]
        qp: list[Any] = []
        if params.symbol:
            conditions.append("symbol = ?")
            qp.append(params.symbol.upper())
        if params.since_date:
            conditions.append("logged_at >= ?")
            qp.append(params.since_date)
        if params.session:
            conditions.append("session = ?")
            qp.append(params.session)
        where = " AND ".join(conditions)

        # Fetch ALL matching rows for accurate aggregate statistics (no LIMIT).
        all_rows = conn.execute(
            f"SELECT outcome, pnl FROM trades WHERE {where} ORDER BY logged_at DESC",
            qp,
        ).fetchall()

        # Separately fetch a capped list for the recent_trades preview.
        recent_rows = conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY logged_at DESC LIMIT ?",
            qp + [params.recent_limit],
        ).fetchall()
        conn.close()

        closed  = [dict(r) for r in all_rows]
        wins    = [t for t in closed if t["outcome"] == "win"]
        losses  = [t for t in closed if t["outcome"] == "loss"]
        scratch = [t for t in closed if t["outcome"] == "scratch"]

        def _avg_pnl(lst: list[dict]) -> float | None:
            vals = [t["pnl"] for t in lst if t.get("pnl") is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        all_pnl  = [t["pnl"] for t in closed if t.get("pnl") is not None]
        win_pnl  = [t["pnl"] for t in wins   if t.get("pnl") is not None]
        loss_pnl = [t["pnl"] for t in losses  if t.get("pnl") is not None]
        gross_profit = sum(p for p in win_pnl  if p > 0)
        gross_loss   = abs(sum(p for p in loss_pnl if p < 0))
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None

        # Current streak — walk from most recent backward.
        streak_count = 0
        streak_type: str | None = None
        for t in closed:  # already DESC ordered
            if t["outcome"] in ("win", "loss"):
                if streak_type is None:
                    streak_type = t["outcome"]
                    streak_count = 1
                elif t["outcome"] == streak_type:
                    streak_count += 1
                else:
                    break

        return json.dumps({
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "scratch": len(scratch),
            "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else None,
            "avg_pnl": _avg_pnl(closed),
            "avg_win": _avg_pnl(wins),
            "avg_loss": _avg_pnl(losses),
            "total_pnl": round(sum(all_pnl), 4) if all_pnl else None,
            "max_win": max(win_pnl) if win_pnl else None,
            "max_loss": min(loss_pnl) if loss_pnl else None,
            "profit_factor": profit_factor,
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "current_streak": {"type": streak_type, "count": streak_count} if streak_type else None,
            "filters": {
                "symbol": params.symbol or "all",
                "since": params.since_date or "all",
                "session": params.session or "all",
            },
            "recent_trades": [dict(r) for r in recent_rows],
        })
    except Exception as exc:
        return _err(str(exc))


# ===========================================================================
# SECTION 8 — SYSTEM INFO  (new)
# ===========================================================================

@mcp.tool(
    name="system_info",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def system_info(
    include_processes: Annotated[bool, "Include top 10 CPU processes (default false)"] = False,
) -> str:
    """Return operating system, CPU, memory, and disk usage statistics.

    Args:
        include_processes: Also return top 10 processes by CPU usage (default false).
    Returns:
        JSON with keys: os, cpu, memory, disk, uptime_hours, (processes if requested).
    """
    try:
        import platform
        result: dict[str, Any] = {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            },
        }
        try:
            import psutil  # type: ignore
            cpu_freq = psutil.cpu_freq()
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            result["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "usage_percent": psutil.cpu_percent(interval=0.5),
                "freq_mhz": round(cpu_freq.current, 1) if cpu_freq else None,
            }
            result["memory"] = {
                "total_gb": round(vm.total / 1e9, 2),
                "used_gb": round(vm.used / 1e9, 2),
                "available_gb": round(vm.available / 1e9, 2),
                "usage_percent": vm.percent,
            }
            result["disk"] = {
                "total_gb": round(disk.total / 1e9, 2),
                "used_gb": round(disk.used / 1e9, 2),
                "free_gb": round(disk.free / 1e9, 2),
                "usage_percent": disk.percent,
            }
            result["uptime_hours"] = round((datetime.datetime.now().timestamp() - psutil.boot_time()) / 3600, 1)
            if include_processes:
                procs = []
                for p in sorted(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]),
                                 key=lambda x: x.info["cpu_percent"] or 0, reverse=True)[:10]:
                    procs.append(p.info)
                result["top_processes"] = procs
        except ImportError:
            result["note"] = "Install psutil for CPU/memory/disk stats: pip install psutil"
        return json.dumps(result)
    except Exception as exc:
        return _err(str(exc))


# ===========================================================================
# SECTION 9 — N8N WORKFLOW AUTOMATION  (existing revamped + 1 new)
# ===========================================================================

@mcp.tool(
    name="n8n_list_workflows",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def n8n_list_workflows(
    active_only: Annotated[bool, "Only return active workflows (default false)"] = False,
) -> str:
    """List all workflows in the local n8n instance. Requires n8n enabled in settings.json.

    Args:
        active_only: Filter to active workflows only (default false).
    Returns:
        JSON with keys: workflows (list of {id, name, active, updated_at}), count.
    """
    s = _n8n_settings()
    if not s["n8n_enabled"]:
        return _err("n8n is disabled. Enable it in settings.json under 'n8n_enabled': true.")
    try:
        import httpx
        resp = httpx.get(f"{s['n8n_url'].rstrip('/')}/api/v1/workflows",
                         headers=_n8n_headers(s), timeout=10)
        resp.raise_for_status()
        workflows = resp.json().get("data", [])
        if active_only:
            workflows = [w for w in workflows if w.get("active")]
        return json.dumps({
            "workflows": [{"id": w.get("id"), "name": w.get("name"),
                           "active": w.get("active"), "updated_at": w.get("updatedAt")} for w in workflows],
            "count": len(workflows),
        })
    except Exception as exc:
        return _err(str(exc), hint=f"Ensure n8n is running at {s['n8n_url']}")


@mcp.tool(
    name="n8n_workflow_detail",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def n8n_workflow_detail(
    workflow_id: Annotated[str, "The n8n workflow ID to retrieve full details for"],
) -> str:
    """Get full details of a single n8n workflow including all nodes and connections.

    Args:
        workflow_id: The numeric or string ID of the n8n workflow.
    Returns:
        JSON with workflow name, active state, nodes (list of {type, name, parameters}), created/updated timestamps.
    """
    s = _n8n_settings()
    if not s["n8n_enabled"]:
        return _err("n8n is disabled. Enable it in settings.json under 'n8n_enabled': true.")
    try:
        import httpx
        resp = httpx.get(f"{s['n8n_url'].rstrip('/')}/api/v1/workflows/{workflow_id}",
                         headers=_n8n_headers(s), timeout=10)
        if resp.status_code == 404:
            return _err(f"Workflow '{workflow_id}' not found. Use n8n_list_workflows to see available IDs.")
        resp.raise_for_status()
        data = resp.json()
        nodes = [{"name": n.get("name"), "type": n.get("type"), "disabled": n.get("disabled", False)}
                 for n in data.get("nodes", [])]
        return json.dumps({
            "id": data.get("id"),
            "name": data.get("name"),
            "active": data.get("active"),
            "nodes": nodes,
            "node_count": len(nodes),
            "created_at": data.get("createdAt"),
            "updated_at": data.get("updatedAt"),
        })
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="n8n_trigger_workflow",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def n8n_trigger_workflow(
    workflow_id: Annotated[str, "The workflow ID to trigger via its Webhook node"],
    payload: Annotated[str, "JSON payload to send (optional, default '{}')"] = "{}",
) -> str:
    """Trigger an n8n workflow via its Webhook trigger node.

    Args:
        workflow_id: Workflow ID (must have a Webhook trigger node).
        payload: JSON string body sent to the webhook (default '{}').
    Returns:
        JSON with keys: workflow_id, status, response.
    """
    s = _n8n_settings()
    if not s["n8n_enabled"]:
        return _err("n8n is disabled.")
    try:
        import httpx
        data = json.loads(payload) if payload else {}
        resp = httpx.post(
            f"{s['n8n_url'].rstrip('/')}/webhook/{workflow_id}",
            json=data, headers=_n8n_headers(s), timeout=30,
        )
        return json.dumps({"workflow_id": workflow_id, "status": resp.status_code, "response": resp.text[:2000]})
    except Exception as exc:
        return _err(str(exc), workflow_id=workflow_id)


@mcp.tool(
    name="n8n_get_executions",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def n8n_get_executions(
    workflow_id: Annotated[str, "Filter executions by workflow ID (empty = all workflows)"] = "",
    limit: Annotated[int, "Number of executions to return (default 5)"] = 5,
) -> str:
    """Retrieve recent workflow execution history from n8n.

    Args:
        workflow_id: Optional filter by workflow ID.
        limit: Number of executions to return (default 5).
    Returns:
        JSON with keys: executions (list of {id, status, startedAt, workflowId}), count.
    """
    s = _n8n_settings()
    if not s["n8n_enabled"]:
        return _err("n8n is disabled.")
    try:
        import httpx
        req_params: dict[str, Any] = {"limit": limit}
        if workflow_id:
            req_params["workflowId"] = workflow_id
        resp = httpx.get(f"{s['n8n_url'].rstrip('/')}/api/v1/executions",
                         params=req_params, headers=_n8n_headers(s), timeout=10)
        resp.raise_for_status()
        execs = resp.json().get("data", [])
        return json.dumps({
            "executions": [{"id": e.get("id"), "status": e.get("status"),
                            "startedAt": e.get("startedAt"), "workflowId": e.get("workflowId")} for e in execs],
            "count": len(execs),
        })
    except Exception as exc:
        return _err(str(exc))


# ===========================================================================
# SECTION 10 — NEW TOOLS: file_copy, date_arithmetic, json_to_csv
# ===========================================================================

class FileCopyInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    source: str = Field(..., description="Source file path relative to the configured file root")
    destination: str = Field(..., description="Destination path relative to the configured file root")
    overwrite: bool = Field(default=False, description="Overwrite destination if it already exists (default false)")


@mcp.tool(
    name="file_copy",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def file_copy(params: FileCopyInput) -> str:
    """Copy a file to a new location within allowed directories.

    The source is left untouched. Parent directories of the destination are
    created automatically.

    Args:
        params.source: Current file path.
        params.destination: Destination path.
        params.overwrite: Allow overwriting an existing destination (default false).
    Returns:
        JSON — source, destination, size_bytes, status.
    """
    ok_src, src = _check_fs_access(params.source)
    if not ok_src:
        return _err(src)
    ok_dst, dst = _check_fs_access(params.destination)
    if not ok_dst:
        return _err(dst)
    if not Path(src).exists():
        return _err(f"Source file does not exist: {src}")
    if not Path(src).is_file():
        return _err(f"Source path is a directory, not a file: {src}")
    if Path(dst).exists() and Path(dst).is_dir():
        return _err(f"Destination path is a directory, not a file: {dst}")
    if Path(dst).exists() and not params.overwrite:
        return _err(f"Destination already exists: {dst}. Set overwrite=true to replace it.")
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = Path(dst).stat().st_size
        return json.dumps({"source": src, "destination": dst, "size_bytes": size, "status": "copied"})
    except Exception as exc:
        return _err(str(exc))


class DateArithmeticInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    date: str = Field(
        default="",
        description="Base date YYYY-MM-DD (or ISO-8601). Leave empty for today.",
    )
    operation: str = Field(default="add", description="'add' or 'subtract'")
    days: int = Field(default=0, description="Days to add/subtract", ge=0)
    weeks: int = Field(default=0, description="Weeks to add/subtract", ge=0)
    months: int = Field(default=0, description="Months to add/subtract", ge=0)
    years: int = Field(default=0, description="Years to add/subtract", ge=0)
    output_format: str = Field(
        default="%Y-%m-%d",
        description="strftime format string for the result, e.g. '%Y-%m-%d' or '%B %d, %Y'",
    )

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        if v not in ("add", "subtract"):
            raise ValueError("operation must be 'add' or 'subtract'")
        return v


@mcp.tool(
    name="date_arithmetic",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def date_arithmetic(params: DateArithmeticInput) -> str:
    """Add or subtract years, months, weeks, and days from any date.

    Useful for planning combine resets, option expirations, trading session
    schedules, or any date-based calculation.

    Args:
        params.date: Base date (YYYY-MM-DD, defaults to today).
        params.operation: 'add' or 'subtract'.
        params.years / months / weeks / days: Duration to apply.
        params.output_format: strftime format for the result (default '%Y-%m-%d').
    Returns:
        JSON — base_date, operation, delta, result_date, result_formatted,
               day_of_week, days_from_today.
    """
    import calendar as _cal
    try:
        base = datetime.date.fromisoformat(params.date[:10]) if params.date.strip() else datetime.date.today()
    except ValueError:
        return _err(f"Cannot parse date '{params.date}'. Use YYYY-MM-DD format.")

    sign = 1 if params.operation == "add" else -1

    # Apply years and months (timedelta doesn't handle calendar months)
    y = base.year + sign * params.years
    m = base.month + sign * params.months
    while m > 12:
        m -= 12
        y += 1
    while m < 1:
        m += 12
        y -= 1
    d = min(base.day, _cal.monthrange(y, m)[1])
    result = datetime.date(y, m, d) + datetime.timedelta(days=sign * (params.weeks * 7 + params.days))

    today = datetime.date.today()
    try:
        formatted = result.strftime(params.output_format)
    except Exception:
        formatted = result.isoformat()

    return json.dumps({
        "base_date": base.isoformat(),
        "operation": params.operation,
        "delta": {"years": params.years, "months": params.months, "weeks": params.weeks, "days": params.days},
        "result_date": result.isoformat(),
        "result_formatted": formatted,
        "day_of_week": result.strftime("%A"),
        "days_from_today": (result - today).days,
    })


class JsonToCsvInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    json_string: str = Field(
        ...,
        description="JSON array of flat objects, e.g. '[{\"name\":\"Alice\",\"score\":95}]'",
        min_length=2,
    )
    delimiter: str = Field(default=",", description="Column delimiter (default ',')", max_length=1)
    include_header: bool = Field(default=True, description="Include header row (default true)")


@mcp.tool(
    name="json_to_csv",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def json_to_csv(params: JsonToCsvInput) -> str:
    """Convert a JSON array of objects to CSV text. Inverse of csv_to_json.

    Column order follows the keys of the first object. Missing keys in
    subsequent rows are output as empty strings.

    Args:
        params.json_string: JSON array of flat objects.
        params.delimiter: Column delimiter (default ',').
        params.include_header: Write column names as first row (default true).
    Returns:
        JSON — csv (string), columns, row_count.
    """
    try:
        data = json.loads(params.json_string)
    except json.JSONDecodeError as exc:
        return _err(f"Invalid JSON: {exc}")
    if not isinstance(data, list):
        return _err("Input must be a JSON array, not a single object or scalar.")
    if not data:
        return json.dumps({"csv": "", "columns": [], "row_count": 0})
    if not isinstance(data[0], dict):
        return _err("Array elements must be JSON objects (dicts).")

    # Preserve first-object key order, then add any extra keys from later rows
    columns: list[str] = []
    seen: set[str] = set()
    for row in data:
        if isinstance(row, dict):
            for k in row:
                if k not in seen:
                    columns.append(k)
                    seen.add(k)

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=params.delimiter)
    if params.include_header:
        writer.writerow(columns)
    for row in data:
        writer.writerow([row.get(col, "") if isinstance(row, dict) else "" for col in columns])

    return json.dumps({"csv": buf.getvalue(), "columns": columns, "row_count": len(data)})


# ===========================================================================
# SECTION 11 — VECTOR MEMORY TOOLS (LangChain + Chroma)
# ===========================================================================

@mcp.tool(
    name="vector_list_collections",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vector_list_collections() -> str:
    """List persistent Chroma collections available to MCP agents."""
    try:
        return json.dumps(chroma_list_collections())
    except VectorStoreError as exc:
        return _err(
            str(exc),
            hint="Install the new vector dependencies from requirements.txt and ensure an Ollama embedding model is available.",
        )
    except Exception as exc:
        return _err(str(exc))


@mcp.tool(
    name="vector_store_text",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def vector_store_text(
    collection_name: Annotated[str, "Target Chroma collection name"],
    text: Annotated[str, "Text content to chunk and index into the vector store"],
    document_id: Annotated[str, "Optional stable document id for this text"] = "",
    metadata_json: Annotated[str, "Optional metadata as a JSON object"] = "{}",
    chunk_size: Annotated[int, "Chunk size in characters (default 1000)"] = 1000,
    chunk_overlap: Annotated[int, "Chunk overlap in characters (default 200)"] = 200,
    embedding_model: Annotated[str, "Optional Ollama embedding model override (default 'nomic-embed-text')"] = "",
) -> str:
    """Index free-form text into the persistent Chroma vector store."""
    ok, metadata = _parse_json_object(metadata_json, "metadata_json")
    if not ok:
        return _err(str(metadata))
    try:
        return json.dumps(chroma_add_text(
            collection_name,
            text,
            document_id=document_id,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            source="inline_text",
        ))
    except VectorStoreError as exc:
        return _err(str(exc), collection_name=collection_name)
    except Exception as exc:
        return _err(str(exc), collection_name=collection_name)


@mcp.tool(
    name="vector_store_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def vector_store_file(
    collection_name: Annotated[str, "Target Chroma collection name"],
    path: Annotated[str, "Text file path to read and index, relative to the configured file root or absolute"],
    document_id: Annotated[str, "Optional stable document id for this file"] = "",
    metadata_json: Annotated[str, "Optional metadata as a JSON object"] = "{}",
    chunk_size: Annotated[int, "Chunk size in characters (default 1000)"] = 1000,
    chunk_overlap: Annotated[int, "Chunk overlap in characters (default 200)"] = 200,
    embedding_model: Annotated[str, "Optional Ollama embedding model override (default 'nomic-embed-text')"] = "",
) -> str:
    """Read a text file and index it into the persistent Chroma vector store."""
    ok_path, resolved = _check_fs_access(path)
    if not ok_path:
        return _err(resolved)
    ok_meta, metadata = _parse_json_object(metadata_json, "metadata_json")
    if not ok_meta:
        return _err(str(metadata))
    try:
        file_text = Path(resolved).read_text(encoding="utf-8", errors="replace")
        metadata = dict(metadata)
        metadata.setdefault("path", resolved)
        return json.dumps(chroma_add_text(
            collection_name,
            file_text,
            document_id=document_id or Path(resolved).stem,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            source=resolved,
        ))
    except VectorStoreError as exc:
        return _err(str(exc), collection_name=collection_name, path=resolved)
    except Exception as exc:
        return _err(str(exc), collection_name=collection_name, path=resolved)


@mcp.tool(
    name="vector_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vector_search(
    collection_name: Annotated[str, "Chroma collection name to search"],
    query: Annotated[str, "Semantic search query"],
    max_results: Annotated[int, "Maximum matches to return (default 5, max 20)"] = 5,
    filter_json: Annotated[str, "Optional metadata filter as a JSON object"] = "{}",
    embedding_model: Annotated[str, "Optional Ollama embedding model override (default 'nomic-embed-text')"] = "",
) -> str:
    """Run semantic search against the persistent Chroma vector store."""
    ok_filter, metadata_filter = _parse_json_object(filter_json, "filter_json")
    if not ok_filter:
        return _err(str(metadata_filter))
    try:
        return json.dumps(chroma_search(
            collection_name,
            query,
            k=max(1, min(20, max_results)),
            metadata_filter=metadata_filter,
            embedding_model=embedding_model,
        ))
    except VectorStoreError as exc:
        return _err(str(exc), collection_name=collection_name, query=query)
    except Exception as exc:
        return _err(str(exc), collection_name=collection_name, query=query)


@mcp.tool(
    name="vector_delete_documents",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vector_delete_documents(
    collection_name: Annotated[str, "Chroma collection name"],
    ids_json: Annotated[str, "JSON array of vector document ids to delete"] = "[]",
) -> str:
    """Delete specific indexed chunks by id from a Chroma collection."""
    ok_ids, ids = _parse_json_array(ids_json, "ids_json")
    if not ok_ids:
        return _err(str(ids))
    try:
        return json.dumps(chroma_delete_document_ids(
            collection_name,
            [str(item) for item in ids],
        ))
    except VectorStoreError as exc:
        return _err(str(exc), collection_name=collection_name)
    except Exception as exc:
        return _err(str(exc), collection_name=collection_name)


@mcp.tool(
    name="vector_delete_collection",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vector_delete_collection(
    collection_name: Annotated[str, "Chroma collection name to permanently delete"],
) -> str:
    """Delete an entire Chroma collection and all of its vectors."""
    try:
        return json.dumps(chroma_delete_collection(collection_name))
    except VectorStoreError as exc:
        return _err(str(exc), collection_name=collection_name)
    except Exception as exc:
        return _err(str(exc), collection_name=collection_name)


# ===========================================================================
# RESOURCE — server manifest  (FIX [10]: count computed dynamically)
# ===========================================================================

_flatten_registered_model_tools()

_ALL_TOOLS: dict[str, list[str]] = {
    "core_utility":   ["get_current_time", "calculator", "word_count", "unit_converter"],
    "text_and_data":  [
        "json_format", "json_to_csv", "csv_to_json", "regex_match", "text_hash",
        "base64_codec", "text_diff", "generate_uuid", "random_data",
        "url_encode_decode", "date_arithmetic",
    ],
    "file_system":    ["list_files", "read_file", "create_file", "write_file",
                       "file_copy", "file_move", "file_delete", "directory_create"],
    "sql_database":   ["sql_create_database", "sql_list_tables", "sql_describe_table",
                       "sql_query", "sql_execute",
                       "sql_export_csv", "sql_import_csv", "sql_backup"],
    "web":            ["web_search", "web_fetch", "extract_text",
                       "http_request", "url_info", "web_news"],
    "email":          ["send_email"],
    "trading":        ["market_quote", "topstepx_status",
                       "trade_journal_add", "trade_journal_stats"],
    "system":         ["system_info"],
    "n8n_automation": ["n8n_list_workflows", "n8n_workflow_detail",
                       "n8n_trigger_workflow", "n8n_get_executions"],
    "vector_memory":  ["vector_list_collections", "vector_store_text", "vector_store_file",
                       "vector_search", "vector_delete_documents", "vector_delete_collection"],
}
_TOTAL_TOOLS = sum(len(v) for v in _ALL_TOOLS.values())


@mcp.resource("info://server")
def server_info() -> str:
    """Return server metadata and a full tool manifest."""
    return json.dumps({
        "name": "local_tools_mcp",
        "version": "4.4.0",
        "total_tools": _TOTAL_TOOLS,     # FIX [10]: now computed, never stale
        "sections": _ALL_TOOLS,
        "bugs_fixed_in_4_1": [
            "[1] path traversal: _under() startswith replaced with boundary-safe _path_under()",
            "[2] get_current_time: params.timezone mutation replaced with local variable",
            "[3] generate_uuid: versions 2/3 silently used uuid4; field now restricted to 1|4",
            "[4] trade_journal_add: PnL 0.0 stored as NULL; pnl field is now Optional[float]",
            "[5] trade_journal_stats: stats computed on limited sample; now uses unlimited query",
            "[6] read_file: false truncation on exact max_chars file; uses peek-read instead",
            "[7] sql_backup: connection leak on exception; fixed with try/finally",
            "[8] random_data: type validated inside loop; moved before loop with early return",
            "[9] http_request: any string accepted as method; Pydantic validator added",
            "[10] server_info: total_tools hardcoded as 44; now computed from _ALL_TOOLS dict",
        ],
        "bugs_fixed_in_4_2": [
            "[11] Flattened BaseModel-backed tool schemas so agents call tools with top-level args instead of nested params",
            "[12] file write targets now reject directories more clearly and report configured-root-relative paths",
            "[13] read_file now reports total_lines for the full file even when content is truncated",
            "[14] file_move/file_copy now reject directory targets when the tool contract is file-only",
        ],
        "bugs_fixed_in_4_3": [
            "[15] Relative file paths now resolve from the configured file root instead of the MCP workspace",
            "[16] Added create_file as a direct, safe primitive for first-time file creation",
            "[17] Rewrote write_file to use simple overwrite/append semantics instead of a mode enum",
            "[18] File tool descriptions now consistently describe configured-root-relative paths for agents",
        ],
        "bugs_fixed_in_4_4": [
            "[19] MCP shutdown now handles Ctrl+C cleanly instead of surfacing an abrupt KeyboardInterrupt crash",
            "[20] Added persistent LangChain Chroma vector memory tools backed by local Ollama embeddings",
        ],
        "new_in_4_1": ["file_copy", "date_arithmetic", "json_to_csv"],
        "new_in_4_3": ["create_file"],
        "new_in_4_4": ["vector_list_collections", "vector_store_text", "vector_store_file", "vector_search"],
    })


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _env_opt(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


# SECURITY FIX [H-1]: The previous implementation hard-coded
# http://localhost:{7860,3000} into the CORS allowlist even when the
# operator set MCP_ALLOWED_ORIGINS. That let any local page on those
# ports drive the full MCP tool surface. We now only include the loopback
# defaults when MCP_ALLOWED_ORIGINS is entirely unset, so an explicit
# setting is always honoured exactly.
_LOOPBACK_BROWSER_ORIGINS = (
    "http://localhost:7860",
    "http://127.0.0.1:7860",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def _parse_allowed_origins(raw: str) -> list[str]:
    entries = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if "*" in entries:
        return ["*"]
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _browser_allowed_origins() -> list[str]:
    raw = os.getenv("MCP_ALLOWED_ORIGINS", "")
    configured = _parse_allowed_origins(raw)
    # SECURITY FIX [H-1]: Refuse to start with a wildcard origin. Wildcards
    # would let any public site on the internet drive every MCP tool.
    if configured == ["*"]:
        raise RuntimeError(
            "MCP_ALLOWED_ORIGINS='*' is unsafe. Set explicit origins "
            "(e.g. 'https://hivebyte.net'); refusing to start."
        )
    if configured:
        return configured
    # No explicit configuration: fall back to the loopback defaults so
    # local-only development still works.
    return list(_LOOPBACK_BROWSER_ORIGINS)


class BrowserAccessHeadersMiddleware:
    """Add headers browsers may require for public-origin -> local-service access.

    SECURITY FIX [H-1 / M-1]: The previous implementation sent
    ``Access-Control-Allow-Private-Network: true`` unconditionally, which
    lowered Chrome's Private Network Access protection for *every*
    request, including ones from origins we never intended to trust. We
    now emit the header only when the request's Origin is in the
    allowlist, so PNA enforcement remains for everyone else.
    """

    def __init__(self, app, allowed_origins: tuple[str, ...] = ()):
        self.app = app
        self.allowed = {o.rstrip("/").lower() for o in allowed_origins}

    async def __call__(self, scope, receive, send):
        request_origin = ""
        if scope.get("type") == "http":
            for name, value in scope.get("headers", []):
                if name == b"origin":
                    try:
                        request_origin = value.decode("latin-1").rstrip("/").lower()
                    except Exception:
                        request_origin = ""
                    break

        allow_pna = request_origin != "" and request_origin in self.allowed

        async def send_wrapper(message):
            if message["type"] == "http.response.start" and allow_pna:
                headers = MutableHeaders(raw=message["headers"])
                headers["Access-Control-Allow-Private-Network"] = "true"
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _build_http_middleware() -> list[Middleware]:
    allowed_origins = _browser_allowed_origins()
    middleware: list[Middleware] = [
        Middleware(
            BrowserAccessHeadersMiddleware,
            allowed_origins=tuple(allowed_origins),
        )
    ]
    if allowed_origins:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=False,
                # SECURITY FIX [M-1]: Tighten from "*" to the exact set of
                # methods/headers the MCP protocol actually needs.
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=[
                    "content-type",
                    "accept",
                    "authorization",
                    "mcp-session-id",
                    "x-requested-with",
                ],
                expose_headers=["Mcp-Session-Id"],
                max_age=600,
            )
        )
    return middleware

if __name__ == "__main__":
    runtime_settings = load_settings_snapshot()
    port = int(os.getenv("MCP_PORT", str(runtime_settings.get("mcp_port", 9000))))
    # SECURITY FIX [L-1]: Default the MCP bind host to loopback. The
    # browser-side code in hosted mode talks to the server from
    # 127.0.0.1/localhost, so this doesn't regress the hosted UX while
    # removing "bound to 0.0.0.0 by default" as an attack surface.
    host = os.getenv("MCP_HOST", str(runtime_settings.get("mcp_host", "127.0.0.1")))
    ssl_certfile = _env_opt("MCP_SSL_CERTFILE")
    ssl_keyfile = _env_opt("MCP_SSL_KEYFILE")
    ssl_keyfile_password = _env_opt("MCP_SSL_KEYFILE_PASSWORD")

    uvicorn_config: dict[str, str] = {}
    if ssl_certfile or ssl_keyfile:
        if not (ssl_certfile and ssl_keyfile):
            raise RuntimeError("Set both MCP_SSL_CERTFILE and MCP_SSL_KEYFILE to enable HTTPS.")
        uvicorn_config["ssl_certfile"] = ssl_certfile
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        if ssl_keyfile_password:
            uvicorn_config["ssl_keyfile_password"] = ssl_keyfile_password

    scheme = "https" if uvicorn_config else "http"
    allowed_origins = _browser_allowed_origins()
    if allowed_origins == ["*"]:
        cors_summary = "*"
    elif allowed_origins:
        cors_summary = ", ".join(allowed_origins)
    else:
        cors_summary = "(same-origin only)"
    app = mcp.http_app(
        path="/mcp",
        transport="streamable-http",
        middleware=_build_http_middleware(),
    )
    print(
        f"local_tools_mcp v4.4.0 - {_TOTAL_TOOLS} tools - {scheme}://{host}:{port}"
        f" - browser origins: {cors_summary}"
    )
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            **uvicorn_config,
        )
    except KeyboardInterrupt:
        print("\nShutting down local_tools_mcp gracefully...")
        log.info("local_tools_mcp shutdown requested by Ctrl+C.")
    finally:
        logging.shutdown()
