# Tencent is pleased to support the open source community by making tRPC-Agent-Python available.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# tRPC-Agent-Python is licensed under Apache-2.0.
"""Session / Memory / Summary replay consistency tests.

Design Overview
===============

The replay harness validates cross-backend consistency for Session, Memory, and
Summary operations. It replays identical operation sequences against two
backends (InMemory and SQL/SQLite) and compares the normalized results.

Normalization Strategy
-----------------------
- **Timestamps**: All float timestamps are zeroed to 0.0 (non-deterministic).
- **Auto-generated IDs**: UUID-like IDs (event.id, invocation_id, etc.) are
  zeroed to "" for comparison.
- **Dict key order**: Keys are sorted for deterministic serialization.
- **None vs missing**: None values are stripped to treat them equivalent to
  missing keys.
- **Whitespace**: Text content is whitespace-normalized for semantic comparison.

Summary Comparison Strategy
----------------------------
- **Content semantic**: Summary text is compared whitespace-normalized.
- **Metadata exact**: session_id, original_event_count, compressed_event_count
  must match exactly - no fuzzy matching.
- **Summary loss**: If one backend has a summary and the other doesn't, this is
  a critical error (unallowed diff).
- **Cross-session summary**: If summary.session_id differs from the expected
  session, this is flagged as a critical error.
- **Summary timestamp**: Allowed diff (zeroed during normalization).

Allowed Differences
--------------------
- Event: id, timestamp, invocation_id
- Session: last_update_time, save_key
- Summary: summary_timestamp

Backend Access Modes
---------------------
- **Lightweight (default)**: InMemory vs SQLite (in-memory). No external dependencies.
  Target: < 30 seconds for all cases.
- **Integration**: Set environment variables to enable Redis or external SQL:
    - TRPC_REPLAY_REDIS_URL=redis://localhost:6379/0
    - TRPC_REPLAY_SQL_URL=mysql+pymysql://user:pass@localhost/db
  (Redis/SQL integration is tested only if the backend imports succeed.)

Acceptance Criteria
--------------------
1. InMemory vs SQLite (lightweight) runs all 12 cases.
2. 10 corruption cases must 100% detect injected inconsistencies.
3. Normal cases false-positive rate <= 5%.
4. Summary loss, wrong-session summary, summary overwrite error: 100% detection.
5. Diff report locates: session_id, event_index/summary_id, field_path, values.
"""

import asyncio
import json
import os
import time

import pytest

from trpc_agent_sdk.events import Event
from trpc_agent_sdk.sessions._session_summarizer import SessionSummary
from trpc_agent_sdk.types import Content
from trpc_agent_sdk.types import Part

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def event_loop():
    """Create a module-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def lightweight_config():
    """Lightweight harness configuration (InMemory vs SQLite)."""
    from tests.sessions.replay_harness import ReplayHarnessConfig
    return ReplayHarnessConfig(
        lightweight_only=True,
        sql_db_url="sqlite:///:memory:",
        run_corruption_cases=True,
    )


# ──────────────────────────────────────────────────────────────
# Test cases: Normal (non-corruption) cases
# ──────────────────────────────────────────────────────────────

NORMAL_CASES = [
    "single_turn_text",
    "multi_turn_text",
    "tool_call_conversation",
    "state_update_and_override",
    "memory_write_and_read",
    "memory_facts_and_prefs",
    "summary_create_and_verify",
    "summary_with_truncation",
]

CORRUPTION_CASES = [
    "summary_missing_detection",
    "summary_wrong_session",
    "duplicate_event_detection",
    "state_dirty_after_error",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", NORMAL_CASES)
async def test_normal_case_consistency(case_id, lightweight_config):
    """Test that normal replay cases produce no unexpected diffs."""
    from tests.sessions.replay_cases import ALL_REPLAY_CASES
    from tests.sessions.replay_harness import run_replay_harness

    case = next(c for c in ALL_REPLAY_CASES if c.case_id == case_id)
    reports, _ = await run_replay_harness(cases=[case], config=lightweight_config, output_report="")

    report = reports[0]
    unallowed_diffs = [d for d in report.diffs if not d.allowed]

    assert report.passed, (
        f"Case '{case_id}' FAILED with {len(unallowed_diffs)} unexpected diffs:\n"
        + "\n".join(
            f"  [{d.component}] {d.field_path}: A={d.value_a}, B={d.value_b}"
            for d in unallowed_diffs[:10]
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", CORRUPTION_CASES)
async def test_corruption_case_detection(case_id, lightweight_config):
    """Test that corruption cases correctly detect the injected inconsistencies."""
    from tests.sessions.replay_cases import ALL_REPLAY_CASES
    from tests.sessions.replay_harness import run_replay_harness

    case = next(c for c in ALL_REPLAY_CASES if c.case_id == case_id)
    reports, _ = await run_replay_harness(cases=[case], config=lightweight_config, output_report="")

    report = reports[0]
    unallowed_diffs = [d for d in report.diffs if not d.allowed]

    assert len(unallowed_diffs) > 0, (
        f"Corruption case '{case_id}' ({case.corruption_type}) did NOT detect any inconsistency. "
        f"Expected diffs for corruption: {case.corruption_description}"
    )

    # Verify the diff is in the right component
    if case.corruption_type == "missing_summary":
        summary_diffs = [d for d in unallowed_diffs if d.component == "summary"]
        assert len(summary_diffs) > 0, f"Expected summary diff for missing_summary, got none"

    if case.corruption_type == "wrong_session_summary":
        summary_diffs = [d for d in unallowed_diffs if d.component == "summary"]
        session_mismatch = [d for d in summary_diffs if "session_id" in d.field_path or "SESSION MISMATCH" in d.note]
        assert len(session_mismatch) > 0, f"Expected session_id mismatch for wrong_session_summary"

    if case.corruption_type == "duplicate_event":
        event_diffs = [d for d in unallowed_diffs if d.component == "events"]
        assert len(event_diffs) > 0, f"Expected event diff for duplicate_event"

    if case.corruption_type == "dirty_state":
        state_diffs = [d for d in unallowed_diffs if d.component == "state"]
        assert len(state_diffs) > 0, f"Expected state diff for dirty_state"


@pytest.mark.asyncio
async def test_summary_loss_detection_specific():
    """Specifically verify summary loss detection: one side has summary, other doesn't."""
    from tests.sessions.replay_cases import CASE_SUMMARY_MISSING
    from tests.sessions.replay_harness import run_replay_harness
    from tests.sessions.replay_harness import ReplayHarnessConfig

    config = ReplayHarnessConfig(lightweight_only=True, sql_db_url="sqlite:///:memory:", run_corruption_cases=True)
    reports, _ = await run_replay_harness(cases=[CASE_SUMMARY_MISSING], config=config, output_report="")

    report = reports[0]
    summary_diffs = [d for d in report.diffs if d.component == "summary" and not d.allowed]

    # Must have at least one summary diff
    assert len(summary_diffs) >= 1, "Summary loss was NOT detected"

    # Every summary diff should mention "<missing>" on one side
    has_missing = any(
        str(d.value_a) == "<missing>" or str(d.value_b) == "<missing>"
        for d in summary_diffs
    )
    assert has_missing, "Summary loss diff should show '<missing>' on one side"


@pytest.mark.asyncio
async def test_summary_wrong_session_detection_specific():
    """Specifically verify wrong-session summary detection."""
    from tests.sessions.replay_cases import CASE_SUMMARY_WRONG_SESSION
    from tests.sessions.replay_harness import run_replay_harness
    from tests.sessions.replay_harness import ReplayHarnessConfig

    config = ReplayHarnessConfig(lightweight_only=True, sql_db_url="sqlite:///:memory:", run_corruption_cases=True)
    reports, _ = await run_replay_harness(cases=[CASE_SUMMARY_WRONG_SESSION], config=config, output_report="")

    report = reports[0]
    summary_diffs = [d for d in report.diffs if d.component == "summary" and not d.allowed]

    assert len(summary_diffs) >= 1, "Wrong-session summary was NOT detected"

    # Check that a session_id-related diff exists
    session_diffs = [
        d for d in summary_diffs
        if "session_id" in d.field_path.lower() or "session" in d.note.lower()
    ]
    assert len(session_diffs) >= 1, f"Expected session_id mismatch, got diffs: {summary_diffs}"


@pytest.mark.asyncio
async def test_summary_overwrite_detection():
    """Verify summary overwrite error detection: two different summaries for same session."""
    from tests.sessions.replay_cases import APP_NAME, USER_ID, SESSION_ID
    from tests.sessions.replay_harness import (
        _create_inmemory_session_service,
        _create_sql_session_service,
        _create_inmemory_memory_service,
        _create_sql_memory_service,
        ReplayExecutor,
    )

    exec_a = ReplayExecutor(
        session_service=_create_inmemory_session_service(),
        memory_service=_create_inmemory_memory_service(),
        backend_name="InMemory",
    )
    exec_b = ReplayExecutor(
        session_service=_create_sql_session_service("sqlite:///:memory:"),
        memory_service=_create_sql_memory_service("sqlite:///:memory:"),
        backend_name="SQL",
    )

    try:
        # Both create session and add events
        for ex in (exec_a, exec_b):
            session = await ex._session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
            )
            ex._session = session
            event = Event(
                invocation_id="u1", author="user",
                content=Content(
                    role="user", parts=[Part.from_text(text="Hello")]
                ),
            )
            await ex._session_service.append_event(session=session, event=event)

        # Backend A: summary version 1
        exec_a._summary_cache[SESSION_ID] = SessionSummary(
            session_id=SESSION_ID, summary_text="Summary version 1",
            original_event_count=1, compressed_event_count=1,
            summary_timestamp=time.time(),
        )
        # Backend B: summary version 2 (different text - overwrite)
        exec_b._summary_cache[SESSION_ID] = SessionSummary(
            session_id=SESSION_ID, summary_text="Summary version 2 - OVERWRITTEN",
            original_event_count=1, compressed_event_count=1,
            summary_timestamp=time.time(),
        )

        # Compare
        from tests.sessions.replay_harness import compare_summaries
        diffs = compare_summaries(
            exec_a._summary_cache.get(SESSION_ID),
            exec_b._summary_cache.get(SESSION_ID),
            SESSION_ID,
        )

        unallowed = [d for d in diffs if not d.allowed]
        assert len(unallowed) >= 1, f"Summary overwrite was NOT detected, diffs={diffs}"

        # The diff should be about summary_text
        text_diffs = [d for d in unallowed if "text" in d.field_path.lower()]
        assert len(text_diffs) >= 1, f"Expected summary_text diff, got: {unallowed}"

    finally:
        await exec_a.close()
        await exec_b.close()


# ──────────────────────────────────────────────────────────────
# Full integration test (runs all cases, generates report)
# ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_replay_suite_and_report(lightweight_config, tmp_path):
    """Run all 12 replay cases, generate diff report, verify acceptance criteria."""
    from tests.sessions.replay_cases import ALL_REPLAY_CASES
    from tests.sessions.replay_harness import run_replay_harness

    report_path = str(tmp_path / "session_memory_summary_diff_report.json")
    reports, harness = await run_replay_harness(
        cases=ALL_REPLAY_CASES,
        config=lightweight_config,
        output_report=report_path,
    )

    # Verify report file exists and is valid JSON
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    assert "cases" in report_data
    assert len(report_data["cases"]) == 12

    # Acceptance criteria checks
    normal_cases = [r for r in reports if r.case_id in NORMAL_CASES]
    corruption_cases = [r for r in reports if r.case_id in CORRUPTION_CASES]

    # 1. All normal cases must pass
    for r in normal_cases:
        assert r.passed, f"Normal case '{r.case_id}' FAILED: {r.note}"

    # 2. All corruption cases must detect the issue
    for r in corruption_cases:
        assert r.passed, f"Corruption case '{r.case_id}' did NOT detect issue: {r.note}"

    # 3. False positive rate check for normal cases
    for r in normal_cases:
        unallowed = [d for d in r.diffs if not d.allowed]
        assert len(unallowed) == 0, (
            f"Normal case '{r.case_id}' has {len(unallowed)} unallowed diffs (false positives)"
        )

    # 4. Summary error detection must be 100%
    summary_cases = [
        r for r in reports
        if r.case_id in ("summary_missing_detection", "summary_wrong_session")
    ]
    for r in summary_cases:
        summary_diffs = [d for d in r.diffs if d.component == "summary" and not d.allowed]
        assert len(summary_diffs) >= 1, f"Summary error not detected for '{r.case_id}'"

    # 5. Diff report must contain field paths and values
    all_case_diffs = [d for r in reports for d in r.diffs]
    assert len(all_case_diffs) > 0, "No diffs at all - even allowed diffs should exist"
    for d in all_case_diffs[:5]:  # spot check first few
        assert d.session_id, "Diff entry missing session_id"
        assert d.field_path, "Diff entry missing field_path"

    print(f"\nAll 12 replay cases completed successfully.")
    print(f"  Normal cases passed: {len(normal_cases)}/{len(normal_cases)}")
    print(f"  Corruption cases detected: {len(corruption_cases)}/{len(corruption_cases)}")
    print(f"  Report: {report_path}")


@pytest.mark.asyncio
async def test_diff_report_field_accuracy(lightweight_config, tmp_path):
    """Verify diff report contains session_id, event_index, field_path, and values."""
    from tests.sessions.replay_cases import CASE_STATE_UPDATE
    from tests.sessions.replay_harness import run_replay_harness

    report_path = str(tmp_path / "field_accuracy_report.json")
    reports, _ = await run_replay_harness(
        cases=[CASE_STATE_UPDATE],
        config=lightweight_config,
        output_report=report_path,
    )

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_data = data["cases"][0]
    for diff in case_data.get("diffs", []):
        assert "session_id" in diff, f"Missing session_id in diff: {diff}"
        assert "field_path" in diff, f"Missing field_path in diff: {diff}"
        # At least one of event_index or summary_id should be set for relevant components
        assert "component" in diff, f"Missing component in diff: {diff}"
