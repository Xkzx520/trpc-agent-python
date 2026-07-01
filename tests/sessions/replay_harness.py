# Tencent is pleased to support the open source community by making tRPC-Agent-Python available.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# tRPC-Agent-Python is licensed under Apache-2.0.
"""Replay harness for Session / Memory / Summary consistency testing.

Architecture:
  ReplayHarness
    ├── executes ReplayCase against two backends (A and B)
    ├── collects results into BackendResult
    ├── normalizes data via Normalizer pipeline
    ├── compares via Comparator pipeline (events, state, memory, summary)
    └── produces DiffReport entries

Normalization strategy:
  - Timestamps: normalized to 0.0 (non-deterministic across backends)
  - Auto-generated IDs: normalized to "" (event.id, session.id may differ)
  - Serialization field order: dict keys sorted for comparison
  - None vs missing: treated as equivalent where appropriate
  - Summary text: whitespace-normalized semantic comparison

Allowed differences (allowed_diff):
  - timestamp: always differs between backends
  - event.id: auto-generated UUID, differs
  - last_update_time: always differs
  - save_key: backend-dependent format
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from trpc_agent_sdk.abc import MemoryServiceABC
from trpc_agent_sdk.abc import MemoryServiceConfig as MemoryConfig
from trpc_agent_sdk.abc import SearchMemoryResponse
from trpc_agent_sdk.abc import SessionServiceABC
from trpc_agent_sdk.events import Event
from trpc_agent_sdk.log import logger
from trpc_agent_sdk.memory import InMemoryMemoryService
from trpc_agent_sdk.sessions import InMemorySessionService
from trpc_agent_sdk.sessions import Session
from trpc_agent_sdk.sessions import SessionServiceConfig
from trpc_agent_sdk.sessions._session_summarizer import SessionSummary
from trpc_agent_sdk.types import Content
from trpc_agent_sdk.types import EventActions
from trpc_agent_sdk.types import Part
from trpc_agent_sdk.utils import user_key

from .replay_cases import APP_NAME
from .replay_cases import SESSION_ID
from .replay_cases import USER_ID
from .replay_cases import ReplayCase
from .replay_cases import ReplayStep

# ──────────────────────────────────────────────────────────────
# Allowed-diff field paths (normalized to dot-notation)
# ──────────────────────────────────────────────────────────────
_ALLOWED_EVENT_FIELDS = {
    "id",           # auto-generated UUID
    "timestamp",    # backend-dependent clock
    "invocation_id",  # may differ if replay produces new IDs
}

_ALLOWED_SESSION_FIELDS = {
    "last_update_time",  # backend-dependent clock
    "save_key",          # backend-dependent format
}

_ALLOWED_SUMMARY_FIELDS = {
    "summary_timestamp",  # backend-dependent clock
}

# Fields that represent "auto-generated ID" patterns
_ID_FIELD_PATTERNS = {"id", "invocation_id", "request_id", "response_id"}


# ──────────────────────────────────────────────────────────────
# Diff entry and report structures
# ──────────────────────────────────────────────────────────────


@dataclass
class DiffEntry:
    """A single diff between two backends for a specific field."""
    session_id: str = ""
    component: str = ""  # events | state | memory | summary
    event_index: Optional[int] = None  # index in event list, if applicable
    summary_id: Optional[str] = None  # summary identifier
    field_path: str = ""  # dot-separated path to the differing field
    value_a: Any = None  # value from backend A
    value_b: Any = None  # value from backend B
    allowed: bool = False  # whether this diff is expected/allowed
    note: str = ""  # explanation of the difference


@dataclass
class BackendResult:
    """Collected results after replaying a case against one backend."""
    backend_name: str = ""
    session: Optional[Session] = None
    memory_response: Optional[SearchMemoryResponse] = None
    summary: Optional[SessionSummary] = None
    error: Optional[str] = None


@dataclass
class CaseDiffReport:
    """Diff report for a single replay case across two backends."""
    case_id: str = ""
    backend_a: str = ""
    backend_b: str = ""
    diffs: List[DiffEntry] = field(default_factory=list)
    passed: bool = True
    note: str = ""


# ──────────────────────────────────────────────────────────────
# Normalizers
# ──────────────────────────────────────────────────────────────


def _normalize_timestamps(obj: Any, _path: str = "") -> Any:
    """Zero out all float timestamp-like fields for comparison."""
    if isinstance(obj, dict):
        return {k: _normalize_timestamps(v, f"{_path}.{k}" if _path else k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_timestamps(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, float):
        # Heuristic: large floats near epoch are timestamps
        if 1e8 < obj < 1e13:
            return 0.0
        if 0 < obj < 1000000:
            # Small float, check path context
            for kw in ("timestamp", "time", "update_time", "last_update", "created_at"):
                if kw in _path.lower():
                    return 0.0
        return obj
    return obj


def _normalize_ids(obj: Any, _path: str = "") -> Any:
    """Zero out auto-generated ID fields that are UUID-like or backend-dependent."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            full_path = f"{_path}.{k}" if _path else k
            if k.lower() in _ID_FIELD_PATTERNS and isinstance(v, str) and len(v) > 20:
                result[k] = ""
            else:
                result[k] = _normalize_ids(v, full_path)
        return result
    if isinstance(obj, list):
        return [_normalize_ids(v, f"{_path}[{i}]") for i, v in enumerate(obj)]
    return obj


def _sort_dict_keys(obj: Any) -> Any:
    """Sort dict keys for deterministic comparison (handles JSON serialization order)."""
    if isinstance(obj, dict):
        return {k: _sort_dict_keys(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        # Sort lists of dicts by a stable hash of their content
        result = [_sort_dict_keys(v) for v in obj]
        if result and all(isinstance(v, dict) for v in result):
            try:
                result.sort(key=lambda d: json.dumps(d, sort_keys=True, default=str))
            except (TypeError, ValueError):
                pass
        return result
    return obj


def _normalize_none_vs_missing(obj: Any) -> Any:
    """Treat None and missing keys as the same by removing None values."""
    if isinstance(obj, dict):
        return {k: _normalize_none_vs_missing(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_normalize_none_vs_missing(v) for v in obj]
    return obj


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for semantic text comparison."""
    if not text:
        return ""
    return " ".join(text.split())


def normalize_event(event: Event) -> Dict[str, Any]:
    """Normalize an Event for comparison."""
    data = event.model_dump(exclude_none=False, mode="json")
    # Apply the normalization pipeline
    data = _normalize_ids(data)
    data = _normalize_timestamps(data)
    data = _normalize_none_vs_missing(data)
    data = _sort_dict_keys(data)
    return data


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize session state for comparison."""
    # Deep copy to avoid mutating
    result = copy.deepcopy(state or {})
    # Strip temp: prefix keys (they're ephemeral)
    result = {k: v for k, v in result.items() if not k.startswith("temp:")}
    result = _normalize_timestamps(result)
    result = _sort_dict_keys(result)
    return result


def normalize_summary(summary: Optional[SessionSummary]) -> Optional[Dict[str, Any]]:
    """Normalize a SessionSummary for comparison."""
    if summary is None:
        return None
    data = {
        "session_id": summary.session_id,
        "summary_text": _normalize_whitespace(summary.summary_text),
        "original_event_count": summary.original_event_count,
        "compressed_event_count": summary.compressed_event_count,
        "summary_timestamp": 0.0,  # always zeroed
    }
    return data


def normalize_memory(memory: Optional[SearchMemoryResponse]) -> Dict[str, Any]:
    """Normalize a SearchMemoryResponse for comparison."""
    if memory is None:
        return {"memories": []}
    result = {"memories": []}
    for mem in memory.memories:
        entry = {
            "author": _normalize_whitespace(mem.author or ""),
            "timestamp": "",  # always zeroed
            "content_text": "",
        }
        if mem.content and mem.content.parts:
            text_parts = [p.text for p in mem.content.parts if p.text]
            entry["content_text"] = _normalize_whitespace(" ".join(text_parts))
        result["memories"].append(entry)
    result = _sort_dict_keys(result)
    return result


# ──────────────────────────────────────────────────────────────
# Comparators
# ──────────────────────────────────────────────────────────────


def _deep_diff(
    a: Any,
    b: Any,
    path: str = "",
    allowed_fields: Optional[set] = None,
) -> List[Tuple[str, Any, Any, bool]]:
    """Recursively diff two normalized data structures.

    Returns list of (field_path, value_a, value_b, is_allowed).
    """
    if allowed_fields is None:
        allowed_fields = set()
    diffs: List[Tuple[str, Any, Any, bool]] = []

    if type(a) != type(b):
        diffs.append((path, a, b, path in allowed_fields))
        return diffs

    if isinstance(a, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for k in sorted(all_keys):
            sub_path = f"{path}.{k}" if path else k
            if k not in a:
                diffs.append((sub_path, "<missing>", b[k], sub_path in allowed_fields))
            elif k not in b:
                diffs.append((sub_path, a[k], "<missing>", sub_path in allowed_fields))
            else:
                diffs.extend(_deep_diff(a[k], b[k], sub_path, allowed_fields))
    elif isinstance(a, list):
        max_len = max(len(a), len(b))
        for i in range(max_len):
            sub_path = f"{path}[{i}]"
            if i >= len(a):
                diffs.append((sub_path, "<missing>", b[i], sub_path in allowed_fields))
            elif i >= len(b):
                diffs.append((sub_path, a[i], "<missing>", sub_path in allowed_fields))
            else:
                diffs.extend(_deep_diff(a[i], b[i], sub_path, allowed_fields))
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
            diffs.append((path, a, b, path in allowed_fields))
    elif a != b:
        diffs.append((path, a, b, path in allowed_fields))

    return diffs


def compare_events(
    events_a: List[Event],
    events_b: List[Event],
    session_id: str,
) -> List[DiffEntry]:
    """Compare normalized event lists from two backends."""
    diffs: List[DiffEntry] = []
    norm_a = [normalize_event(e) for e in (events_a or [])]
    norm_b = [normalize_event(e) for e in (events_b or [])]

    max_len = max(len(norm_a), len(norm_b))
    for i in range(max_len):
        prefix = f"events[{i}]"
        if i >= len(norm_a):
            diffs.append(DiffEntry(
                session_id=session_id,
                component="events",
                event_index=i,
                field_path=f"{prefix}",
                value_a="<missing>",
                value_b=norm_b[i],
                allowed=False,
                note=f"Event at index {i} missing in backend A",
            ))
        elif i >= len(norm_b):
            diffs.append(DiffEntry(
                session_id=session_id,
                component="events",
                event_index=i,
                field_path=f"{prefix}",
                value_a=norm_a[i],
                value_b="<missing>",
                allowed=False,
                note=f"Event at index {i} missing in backend B",
            ))
        else:
            for field_path, va, vb, allowed in _deep_diff(norm_a[i], norm_b[i], prefix, _ALLOWED_EVENT_FIELDS):
                diffs.append(DiffEntry(
                    session_id=session_id,
                    component="events",
                    event_index=i,
                    field_path=field_path,
                    value_a=va if not isinstance(va, (dict, list)) else json.dumps(va, default=str),
                    value_b=vb if not isinstance(vb, (dict, list)) else json.dumps(vb, default=str),
                    allowed=allowed,
                    note="allowed backend difference" if allowed else "",
                ))
    return diffs


def compare_state(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
    session_id: str,
) -> List[DiffEntry]:
    """Compare normalized state from two backends."""
    diffs: List[DiffEntry] = []
    norm_a = normalize_state(state_a)
    norm_b = normalize_state(state_b)

    for field_path, va, vb, allowed in _deep_diff(norm_a, norm_b, "state", _ALLOWED_SESSION_FIELDS):
        diffs.append(DiffEntry(
            session_id=session_id,
            component="state",
            field_path=field_path,
            value_a=va if not isinstance(va, (dict, list)) else json.dumps(va, default=str),
            value_b=vb if not isinstance(vb, (dict, list)) else json.dumps(vb, default=str),
            allowed=allowed,
            note="allowed backend difference" if allowed else "",
        ))
    return diffs


def compare_summaries(
    summary_a: Optional[SessionSummary],
    summary_b: Optional[SessionSummary],
    session_id: str,
) -> List[DiffEntry]:
    """Compare normalized summaries from two backends.

    Key requirements:
    - Summary loss (one None, one not): MUST be detected
    - Wrong session_id: MUST be detected
    - Text semantic comparison: whitespace-normalized
    - Metadata: exact comparison required (original_event_count, compressed_event_count, session_id)
    """
    diffs: List[DiffEntry] = []

    if summary_a is None and summary_b is None:
        return diffs

    if summary_a is None and summary_b is not None:
        diffs.append(DiffEntry(
            session_id=session_id,
            component="summary",
            summary_id=getattr(summary_b, "summary_text", "")[:50],
            field_path="summary",
            value_a="<missing>",
            value_b=normalize_summary(summary_b),
            allowed=False,
            note="SUMMARY MISSING: Backend A has no summary, Backend B has summary",
        ))
        return diffs

    if summary_a is not None and summary_b is None:
        diffs.append(DiffEntry(
            session_id=session_id,
            component="summary",
            summary_id=getattr(summary_a, "summary_text", "")[:50],
            field_path="summary",
            value_a=normalize_summary(summary_a),
            value_b="<missing>",
            allowed=False,
            note="SUMMARY MISSING: Backend A has summary, Backend B has no summary",
        ))
        return diffs

    # Both exist - normalize and compare
    norm_a = normalize_summary(summary_a)
    norm_b = normalize_summary(summary_b)

    # Special check: session_id mismatch (critical)
    sid_a = summary_a.session_id
    sid_b = summary_b.session_id
    if sid_a != sid_b:
        diffs.append(DiffEntry(
            session_id=session_id,
            component="summary",
            summary_id=sid_a,
            field_path="summary.session_id",
            value_a=sid_a,
            value_b=sid_b,
            allowed=False,
            note="SUMMARY SESSION MISMATCH: summary belongs to different session",
        ))

    # Full diff
    for field_path, va, vb, allowed in _deep_diff(norm_a, norm_b, "summary", _ALLOWED_SUMMARY_FIELDS):
        diffs.append(DiffEntry(
            session_id=session_id,
            component="summary",
            summary_id=sid_a,
            field_path=field_path,
            value_a=va if not isinstance(va, (dict, list)) else json.dumps(va, default=str),
            value_b=vb if not isinstance(vb, (dict, list)) else json.dumps(vb, default=str),
            allowed=allowed,
            note="allowed backend difference" if allowed else "",
        ))

    return diffs


def compare_memory(
    memory_a: Optional[SearchMemoryResponse],
    memory_b: Optional[SearchMemoryResponse],
    session_id: str,
) -> List[DiffEntry]:
    """Compare normalized memory search results."""
    diffs: List[DiffEntry] = []
    norm_a = normalize_memory(memory_a)
    norm_b = normalize_memory(memory_b)

    for field_path, va, vb, allowed in _deep_diff(norm_a, norm_b, "memory"):
        diffs.append(DiffEntry(
            session_id=session_id,
            component="memory",
            field_path=field_path,
            value_a=va if not isinstance(va, (dict, list)) else json.dumps(va, default=str),
            value_b=vb if not isinstance(vb, (dict, list)) else json.dumps(vb, default=str),
            allowed=allowed,
            note="allowed backend difference" if allowed else "",
        ))
    return diffs


# ──────────────────────────────────────────────────────────────
# Backend factory
# ──────────────────────────────────────────────────────────────


def _create_inmemory_session_service() -> InMemorySessionService:
    """Create an InMemory session service."""
    config = SessionServiceConfig()
    config.clean_ttl_config()
    return InMemorySessionService(session_config=config)


def _create_sql_session_service(db_url: str = "sqlite:///:memory:") -> "SqlSessionService":
    """Create a SQL session service (sqlite fallback always available)."""
    from trpc_agent_sdk.sessions._sql_session_service import SqlSessionService
    config = SessionServiceConfig()
    config.clean_ttl_config()
    return SqlSessionService(db_url=db_url, session_config=config, is_async=True)


def _create_inmemory_memory_service() -> InMemoryMemoryService:
    """Create an InMemory memory service."""
    mc = MemoryConfig(enabled=True)
    mc.clean_ttl_config()
    return InMemoryMemoryService(memory_service_config=mc, enabled=True)


def _create_sql_memory_service(db_url: str = "sqlite:///:memory:") -> "SqlMemoryService":
    """Create a SQL memory service."""
    from trpc_agent_sdk.memory._sql_memory_service import SqlMemoryService
    mc = MemoryConfig(enabled=True)
    mc.clean_ttl_config()
    return SqlMemoryService(db_url=db_url, memory_service_config=mc, enabled=True, is_async=True)


# ──────────────────────────────────────────────────────────────
# Replay executor
# ──────────────────────────────────────────────────────────────


class ReplayExecutor:
    """Executes a replay case against a single backend.

    Handles the create_session, append_event, store_memory, create_summary
    operations as defined by the ReplayCase steps.
    """

    def __init__(
        self,
        session_service: SessionServiceABC,
        memory_service: Optional[MemoryServiceABC] = None,
        backend_name: str = "unknown",
    ):
        self._session_service = session_service
        self._memory_service = memory_service
        self._backend_name = backend_name
        self._session: Optional[Session] = None
        # Internal summary cache (simulates SummarizerSessionManager cache)
        self._summary_cache: Dict[str, SessionSummary] = {}
        self._memory_responses: List[SearchMemoryResponse] = []

    @property
    def session(self) -> Optional[Session]:
        return self._session

    async def execute(self, case: ReplayCase) -> BackendResult:
        """Execute all steps of a replay case."""
        self._session = None
        self._summary_cache = {}
        self._memory_responses = []

        for step in case.steps:
            try:
                await self._execute_step(step)
            except Exception as e:
                return BackendResult(
                    backend_name=self._backend_name,
                    error=f"Step '{step.op}' failed: {e}",
                )

        # Get final state by re-reading session
        final_session = None
        if self._session:
            final_session = await self._session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=self._session.id,
            )

        # Get final summary
        final_summary = self._summary_cache.get(SESSION_ID)

        return BackendResult(
            backend_name=self._backend_name,
            session=final_session,
            memory_response=self._memory_responses[-1] if self._memory_responses else None,
            summary=final_summary,
        )

    async def _execute_step(self, step: ReplayStep) -> None:
        """Execute a single replay step."""
        op = step.op
        kwargs = step.kwargs

        if op == "create_session":
            state = kwargs.get("state", None)
            session_id = kwargs.get("session_id", SESSION_ID)
            self._session = await self._session_service.create_session(
                app_name=kwargs.get("app_name", APP_NAME),
                user_id=kwargs.get("user_id", USER_ID),
                state=state,
                session_id=session_id,
            )

        elif op == "append_event":
            event = kwargs["event"]
            state_delta = kwargs.get("state_delta")
            if state_delta:
                # Attach state delta to event actions
                event.actions.state_delta.update(state_delta)
            await self._session_service.append_event(session=self._session, event=event)

        elif op == "update_session":
            await self._session_service.update_session(self._session)

        elif op == "update_session_events":
            # Directly update session events in the backend
            events = kwargs.get("events", [])
            if self._session:
                self._session.events = events
                await self._session_service.update_session(self._session)

        elif op == "store_memory":
            if self._memory_service and self._session:
                await self._memory_service.store_session(self._session)

        elif op == "search_memory":
            if self._memory_service and self._session:
                query = kwargs.get("query", "")
                response = await self._memory_service.search_memory(
                    key=user_key(APP_NAME, USER_ID),
                    query=query,
                    limit=kwargs.get("limit", 10),
                )
                self._memory_responses.append(response)

        elif op == "create_summary":
            summary_text = kwargs["summary_text"]
            original_count = kwargs.get("original_event_count", 0)
            compressed_count = kwargs.get("compressed_event_count", 0)
            keep_recent = kwargs.get("keep_recent_count", 2)

            # Create SessionSummary in cache
            summary = SessionSummary(
                session_id=SESSION_ID,
                summary_text=summary_text,
                original_event_count=original_count,
                compressed_event_count=compressed_count,
                summary_timestamp=time.time(),
            )
            self._summary_cache[SESSION_ID] = summary

            # Simulate event compression: create summary event + truncate
            if self._session:
                # Insert summary event at front
                summary_event = Event(
                    invocation_id="summary",
                    author="system",
                    content=Content(
                        role="user",
                        parts=[Part.from_text(
                            text=f"Previous conversation summary: {summary_text}"
                        )],
                    ),
                    timestamp=time.time(),
                )
                summary_event.set_summary_event(True)

                events = list(self._session.events)
                if keep_recent > 0 and len(events) > keep_recent:
                    # Keep only summary + recent events
                    recent = events[-keep_recent:]
                    # Move old events to historical
                    old_events = events[:-keep_recent]
                    self._session.historical_events.extend(old_events)
                    self._session.events = [summary_event] + recent
                else:
                    self._session.events.insert(0, summary_event)

                await self._session_service.update_session(self._session)

        elif op == "inject_corruption":
            corruption = kwargs.get("corruption_type", "")
            if corruption == "duplicate_event" and self._session and len(self._session.events) > 0:
                # Append a duplicate of the last event
                dup = copy.deepcopy(self._session.events[-1])
                self._session.events.append(dup)
                await self._session_service.update_session(self._session)

            elif corruption == "dirty_state" and self._session:
                # Corrupt a state key
                self._session.state["corrupted_key"] = "bad_data_from_partial_write"
                await self._session_service.update_session(self._session)

            elif corruption == "missing_summary":
                # Remove summary from cache
                self._summary_cache.pop(SESSION_ID, None)
                # Also remove summary events from session
                if self._session:
                    self._session.events = [
                        e for e in self._session.events if not e.is_summary_event()
                    ]
                    await self._session_service.update_session(self._session)

            elif corruption == "wrong_session_summary":
                # Replace with a summary for a different session
                if SESSION_ID in self._summary_cache:
                    self._summary_cache[SESSION_ID] = SessionSummary(
                        session_id="wrong-session-id-999",
                        summary_text=self._summary_cache[SESSION_ID].summary_text,
                        original_event_count=self._summary_cache[SESSION_ID].original_event_count,
                        compressed_event_count=self._summary_cache[SESSION_ID].compressed_event_count,
                        summary_timestamp=time.time(),
                    )

    async def close(self) -> None:
        """Clean up resources."""
        if self._session_service:
            await self._session_service.close()
        if self._memory_service:
            await self._memory_service.close()


# ──────────────────────────────────────────────────────────────
# Replay harness orchestrator
# ──────────────────────────────────────────────────────────────


def _should_skip_backend(env_var: str) -> bool:
    """Check if a backend should be skipped based on environment variable."""
    val = os.environ.get(env_var, "").strip().lower()
    return val in ("0", "false", "no", "skip", "")


@dataclass
class ReplayHarnessConfig:
    """Configuration for the replay harness."""
    lightweight_only: bool = True
    """If True, only run InMemory vs SQLite (always available)."""
    enable_redis: bool = False
    """If True, also test against Redis backend."""
    sql_db_url: str = "sqlite:///:memory:"
    """SQL database URL for the SQL backend."""
    run_corruption_cases: bool = True
    """If True, run cases that inject intentional corruption."""


class ReplayHarness:
    """Orchestrates replay execution and comparison across backends."""

    def __init__(self, config: Optional[ReplayHarnessConfig] = None):
        self._config = config or ReplayHarnessConfig()
        self._all_diffs: List[CaseDiffReport] = []

    @property
    def diff_reports(self) -> List[CaseDiffReport]:
        return self._all_diffs

    async def run_all_cases(self, cases: List[ReplayCase]) -> List[CaseDiffReport]:
        """Run all replay cases and return diff reports."""
        self._all_diffs = []

        for case in cases:
            # Skip corruption cases in lightweight mode? No, we run them all
            # but corruption cases should have expected diffs
            report = await self._run_case(case)
            self._all_diffs.append(report)

        return self._all_diffs

    async def _run_case(self, case: ReplayCase) -> CaseDiffReport:
        """Run a single replay case against two backends."""
        logger.info("Running replay case: %s", case.case_id)

        # ── Backend A: InMemory ──
        exec_a = ReplayExecutor(
            session_service=_create_inmemory_session_service(),
            memory_service=_create_inmemory_memory_service(),
            backend_name="InMemory",
        )

        # ── Backend B: SQL (SQLite) ──
        exec_b = ReplayExecutor(
            session_service=_create_sql_session_service(self._config.sql_db_url),
            memory_service=_create_sql_memory_service(self._config.sql_db_url),
            backend_name="SQL",
        )

        try:
            # Execute case on both backends
            result_a, result_b = await asyncio.gather(
                exec_a.execute(case),
                exec_b.execute(case),
            )

            # ── Optional: inject corruption into backend B ──
            if case.corruption_type and self._config.run_corruption_cases:
                await self._inject_corruption(exec_b, case, result_b)

            # Re-read backend B after corruption injection if needed
            if case.corruption_type and self._config.run_corruption_cases:
                if result_b.session:
                    result_b.session = await exec_b._session_service.get_session(
                        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
                    )

            # ── Compare ──
            report = self._compare_results(case, result_a, result_b)

            # Determine pass/fail
            unallowed_diffs = [d for d in report.diffs if not d.allowed]
            if case.corruption_type:
                # For corruption cases, we expect diffs
                report.passed = len(unallowed_diffs) > 0
                report.note = f"Corruption case '{case.corruption_type}': expected diff detected" if report.passed else \
                              f"FAILED: corruption '{case.corruption_type}' was NOT detected"
            else:
                report.passed = len(unallowed_diffs) == 0
                report.note = "All diffs are allowed" if report.passed else \
                              f"FAILED: {len(unallowed_diffs)} unexpected diffs found"

            return report

        finally:
            await exec_a.close()
            await exec_b.close()

    async def _inject_corruption(
        self,
        executor: ReplayExecutor,
        case: ReplayCase,
        result: BackendResult,
    ) -> None:
        """Inject intentional corruption into backend B for testing."""
        ct = case.corruption_type

        if ct == "duplicate_event" and executor.session and executor.session.events:
            executor.session.events.append(copy.deepcopy(executor.session.events[-1]))
            await executor._session_service.update_session(executor.session)

        elif ct == "dirty_state" and executor.session:
            executor.session.state["config_key"] = "corrupted_value"
            await executor._session_service.update_session(executor.session)

        elif ct == "missing_summary":
            executor._summary_cache.pop(SESSION_ID, None)
            if executor.session:
                executor.session.events = [
                    e for e in executor.session.events if not e.is_summary_event()
                ]
                await executor._session_service.update_session(executor.session)

        elif ct == "wrong_session_summary":
            if SESSION_ID in executor._summary_cache:
                existing = executor._summary_cache[SESSION_ID]
                executor._summary_cache[SESSION_ID] = SessionSummary(
                    session_id="wrong-session-id-999",
                    summary_text=existing.summary_text,
                    original_event_count=existing.original_event_count,
                    compressed_event_count=existing.compressed_event_count,
                    summary_timestamp=time.time(),
                )

    def _compare_results(
        self,
        case: ReplayCase,
        result_a: BackendResult,
        result_b: BackendResult,
    ) -> CaseDiffReport:
        """Compare results from two backends."""
        report = CaseDiffReport(
            case_id=case.case_id,
            backend_a=result_a.backend_name,
            backend_b=result_b.backend_name,
        )

        sid = SESSION_ID

        # Handle errors
        if result_a.error:
            report.diffs.append(DiffEntry(
                session_id=sid, component="error", field_path="backend_a",
                value_a=result_a.error, value_b="", allowed=False,
                note=f"Backend A error: {result_a.error}",
            ))
        if result_b.error:
            report.diffs.append(DiffEntry(
                session_id=sid, component="error", field_path="backend_b",
                value_a="", value_b=result_b.error, allowed=False,
                note=f"Backend B error: {result_b.error}",
            ))
        if result_a.error or result_b.error:
            report.passed = False
            return report

        # Compare events
        events_a = result_a.session.events if result_a.session else []
        events_b = result_b.session.events if result_b.session else []
        report.diffs.extend(compare_events(events_a, events_b, sid))

        # Compare state
        state_a = result_a.session.state if result_a.session else {}
        state_b = result_b.session.state if result_b.session else {}
        report.diffs.extend(compare_state(state_a, state_b, sid))

        # Compare memory
        report.diffs.extend(compare_memory(result_a.memory_response, result_b.memory_response, sid))

        # Compare summary
        summary_a = result_a.summary
        summary_b = result_b.summary
        report.diffs.extend(compare_summaries(summary_a, summary_b, sid))

        return report

    def generate_report_json(self, output_path: str) -> None:
        """Generate a JSON diff report file."""
        report_data = {
            "generated_at": time.time(),
            "backend_a": "InMemory",
            "backend_b": "SQL",
            "total_cases": len(self._all_diffs),
            "cases": [],
            "summary": {
                "passed": 0,
                "failed": 0,
                "total_diffs": 0,
                "unallowed_diffs": 0,
            },
        }

        for case_report in self._all_diffs:
            unallowed = [d for d in case_report.diffs if not d.allowed]
            entry = {
                "case_id": case_report.case_id,
                "passed": case_report.passed,
                "note": case_report.note,
                "total_diffs": len(case_report.diffs),
                "unallowed_diffs": len(unallowed),
                "diffs": [
                    {
                        "session_id": d.session_id,
                        "component": d.component,
                        "event_index": d.event_index,
                        "summary_id": d.summary_id,
                        "field_path": d.field_path,
                        "value_a": str(d.value_a)[:500] if d.value_a is not None else None,
                        "value_b": str(d.value_b)[:500] if d.value_b is not None else None,
                        "allowed": d.allowed,
                        "note": d.note,
                    }
                    for d in case_report.diffs
                ],
            }
            report_data["cases"].append(entry)
            if case_report.passed:
                report_data["summary"]["passed"] += 1
            else:
                report_data["summary"]["failed"] += 1
            report_data["summary"]["total_diffs"] += len(case_report.diffs)
            report_data["summary"]["unallowed_diffs"] += len(unallowed)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Diff report written to %s", output_path)


# ──────────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────────


async def run_replay_harness(
    cases: Optional[List[ReplayCase]] = None,
    config: Optional[ReplayHarnessConfig] = None,
    output_report: str = "session_memory_summary_diff_report.json",
) -> Tuple[List[CaseDiffReport], ReplayHarness]:
    """Run the replay harness and generate a diff report.

    Args:
        cases: List of replay cases to run. If None, runs all cases.
        config: Harness configuration.
        output_report: Path for the JSON diff report.

    Returns:
        Tuple of (diff reports, harness instance).
    """
    if cases is None:
        from .replay_cases import ALL_REPLAY_CASES
        cases = ALL_REPLAY_CASES

    harness = ReplayHarness(config=config)
    reports = await harness.run_all_cases(cases)
    harness.generate_report_json(output_report)
    return reports, harness
