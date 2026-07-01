# Tencent is pleased to support the open source community by making tRPC-Agent-Python available.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# tRPC-Agent-Python is licensed under Apache-2.0.
"""Replay case definitions for Session / Memory / Summary consistency testing.

Each replay case describes a sequence of operations to be executed against
multiple backends, plus expected outcomes for comparison.

Case Index
==========
 1. single_turn_text           - Simple user text + agent text response
 2. multi_turn_text            - Multiple rounds of user/assistant events
 3. tool_call_conversation     - function_call + function_response
 4. state_update_and_override  - Multiple state writes and overwrites
 5. memory_write_and_read      - Store session memory and search
 6. memory_facts_and_prefs     - Simulate user preferences and facts
 7. summary_create_and_verify  - Create summary, verify content/metadata
 8. summary_with_truncation    - Events compressed; summary anchors context
 9. summary_missing_detection  - Inject summary loss for detection
10. summary_wrong_session      - Inject cross-session summary for detection
11. duplicate_event_detection  - Detect duplicate events after simulated re-write
12. state_dirty_after_error    - Partial state corruption detection
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from trpc_agent_sdk.events import Event
from trpc_agent_sdk.types import Content
from trpc_agent_sdk.types import FunctionCall
from trpc_agent_sdk.types import FunctionResponse
from trpc_agent_sdk.types import Part

# ──────────────────────────────────────────────────────────────
# Constants shared across all cases
# ──────────────────────────────────────────────────────────────
APP_NAME = "replay_test_app"
USER_ID = "replay_test_user"
SESSION_ID = "replay-test-session-001"

# ──────────────────────────────────────────────────────────────
# Helper factories
# ──────────────────────────────────────────────────────────────


def _make_user_event(text: str, ts: float = 0.0) -> Event:
    return Event(
        invocation_id="user-inv",
        author="user",
        content=Content(role="user", parts=[Part.from_text(text=text)]),
        timestamp=ts or time.time(),
    )


def _make_agent_text(text: str, ts: float = 0.0) -> Event:
    return Event(
        invocation_id="agent-inv",
        author="agent",
        content=Content(role="model", parts=[Part.from_text(text=text)]),
        timestamp=ts or time.time(),
    )


def _make_tool_call_event(name: str, args: dict, ts: float = 0.0) -> Event:
    fc = FunctionCall(name=name, args=args)
    return Event(
        invocation_id="tool-inv",
        author="agent",
        content=Content(role="model", parts=[Part(function_call=fc)]),
        timestamp=ts or time.time(),
    )


def _make_tool_response_event(name: str, response: dict, ts: float = 0.0) -> Event:
    fr = FunctionResponse(name=name, response=response)
    return Event(
        invocation_id="tool-resp-inv",
        author="agent",
        content=Content(role="user", parts=[Part(function_response=fr)]),
        timestamp=ts or time.time(),
    )


def _make_summary_event(text: str, ts: float = 0.0) -> Event:
    ev = Event(
        invocation_id="summary",
        author="system",
        content=Content(role="user", parts=[Part.from_text(text=f"Previous conversation summary: {text}")]),
        timestamp=ts or time.time(),
    )
    ev.set_summary_event(True)
    return ev


# ──────────────────────────────────────────────────────────────
# ReplayCase container
# ──────────────────────────────────────────────────────────────


@dataclass
class ReplayStep:
    """A single operation in a replay case."""
    op: str  # create_session | append_event | update_session | update_session_events |
    # store_memory | search_memory | create_summary | get_session_summary | inject_corruption
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayCase:
    """Defines a full replay scenario."""
    case_id: str
    description: str
    # Sequence of operations to replay
    steps: List[ReplayStep] = field(default_factory=list)
    # Expected outcomes *before* any intentional corruption
    expected_event_count: int = 0
    expected_state: Dict[str, Any] = field(default_factory=dict)
    expected_memory_hits: int = 0
    expected_summary_available: bool = False
    expected_summary_text: str = ""
    # For corruption-detection cases: what corruption we inject
    corruption_type: Optional[str] = None  # missing_summary | wrong_session_summary | duplicate_event | dirty_state
    corruption_description: str = ""


# ──────────────────────────────────────────────────────────────
# Case 1: single_turn_text
# ──────────────────────────────────────────────────────────────
CASE_SINGLE_TURN = ReplayCase(
    case_id="single_turn_text",
    description="Single turn: user text followed by agent text response",
    expected_event_count=2,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("Hello, how are you?")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("I'm doing well, thank you!")}),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 2: multi_turn_text
# ──────────────────────────────────────────────────────────────
CASE_MULTI_TURN = ReplayCase(
    case_id="multi_turn_text",
    description="Multiple consecutive user/assistant turns",
    expected_event_count=6,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("What's the weather?")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("It's sunny, 25°C.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("And tomorrow?")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Tomorrow will be cloudy, 22°C.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("Thanks!")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("You're welcome!")}),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 3: tool_call_conversation
# ──────────────────────────────────────────────────────────────
CASE_TOOL_CALL = ReplayCase(
    case_id="tool_call_conversation",
    description="Conversation with function_call and function_response events",
    expected_event_count=4,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("Search for Python tutorials")}),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_tool_call_event("web_search", {"query": "Python tutorials", "max_results": 5})
            },
        ),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_tool_response_event("web_search", {"results": ["result1", "result2", "result3"]})
            },
        ),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("I found 3 Python tutorials for you: result1, result2, result3.")
            },
        ),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 4: state_update_and_override
# ──────────────────────────────────────────────────────────────
CASE_STATE_UPDATE = ReplayCase(
    case_id="state_update_and_override",
    description="Multiple state writes: initial, app-level, user-level, session-scoped, override",
    expected_event_count=2,
    expected_state={
        "app:theme": "dark",
        "app:version": "2.0",
        "user:name": "AliceUpdated",
        "user:pref_lang": "zh-CN",
        "session_key": "overridden_value",
        "counter": 2,
    },
    steps=[
        ReplayStep(
            op="create_session",
            kwargs={
                "app_name": APP_NAME,
                "user_id": USER_ID,
                "session_id": SESSION_ID,
                "state": {"app:theme": "dark", "user:name": "Alice", "session_key": "initial"},
            },
        ),
        # First event with state delta
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_user_event("Update my preferences"),
                "state_delta": {"user:pref_lang": "en-US", "counter": 1},
            },
        ),
        # Second event overrides some state
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Preferences updated!"),
                "state_delta": {
                    "user:name": "AliceUpdated",
                    "user:pref_lang": "zh-CN",
                    "app:version": "2.0",
                    "session_key": "overridden_value",
                    "counter": 2,
                },
            },
        ),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 5: memory_write_and_read
# ──────────────────────────────────────────────────────────────
CASE_MEMORY_WRITE_READ = ReplayCase(
    case_id="memory_write_and_read",
    description="Store a session in memory service and search for relevant content",
    expected_event_count=4,
    expected_memory_hits=1,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("My favorite color is blue.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Got it, blue is your favorite!")}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("I live in Shanghai.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Shanghai is a great city!")}),
        ReplayStep(op="store_memory"),
        ReplayStep(op="search_memory", kwargs={"query": "favorite color"}),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 6: memory_facts_and_prefs
# ──────────────────────────────────────────────────────────────
CASE_MEMORY_FACTS = ReplayCase(
    case_id="memory_facts_and_prefs",
    description="Store user facts and preferences in memory, then verify retrieval",
    expected_event_count=6,
    expected_memory_hits=2,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(
            op="append_event",
            kwargs={"event": _make_user_event("I prefer dark mode and use Python for development.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Noted: dark mode preference and Python.")}),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_user_event("My birthday is March 15th and I work as a backend engineer.")
            }),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Got it, birthday March 15th, backend engineer.")
            }),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_user_event("I like swimming and hiking on weekends.")
            }),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Swimming and hiking - great activities!")
            }),
        ReplayStep(op="store_memory"),
        ReplayStep(op="search_memory", kwargs={"query": "Python developer birthday"}),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 7: summary_create_and_verify
# ──────────────────────────────────────────────────────────────
_SUMMARY_TEXT_7 = (
    "User and agent had a conversation about project planning. "
    "Key decisions: use Python for backend, React for frontend, "
    "and PostgreSQL for database. Action items: set up CI/CD pipeline."
)

CASE_SUMMARY_CREATE = ReplayCase(
    case_id="summary_create_and_verify",
    description="Create a summary after conversation, verify content, version, metadata",
    expected_event_count=6,
    expected_summary_available=True,
    expected_summary_text=_SUMMARY_TEXT_7,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event",
                   kwargs={"event": _make_user_event("Let's plan our new project architecture.")}),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Great! Let's start by choosing the tech stack.")
            }),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("I think Python for backend.")}),
        ReplayStep(
            op="append_event",
            kwargs={"event": _make_agent_text("Python is good. For frontend, React is popular.")}),
        ReplayStep(
            op="append_event",
            kwargs={"event": _make_user_event("Let's also use PostgreSQL and set up CI/CD.")}),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Agreed. Summary: Python + React + PostgreSQL + CI/CD.")
            }),
        ReplayStep(
            op="create_summary",
            kwargs={
                "summary_text": _SUMMARY_TEXT_7,
                "original_event_count": 6,
                "compressed_event_count": 3,
            },
        ),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 8: summary_with_event_truncation
# ──────────────────────────────────────────────────────────────
_SUMMARY_TEXT_8 = (
    "Long conversation about travel planning. User wants to visit Japan. "
    "Key details: budget $3000, dates flexible in October, "
    "interested in Tokyo and Kyoto."
)

CASE_SUMMARY_TRUNCATION = ReplayCase(
    case_id="summary_with_truncation",
    description="Summary compresses historical events; remaining events + summary restore context",
    expected_event_count=4,  # after truncation: summary + 2 recent + 1 new
    expected_summary_available=True,
    expected_summary_text=_SUMMARY_TEXT_8,
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        # Build up "long" conversation (8 events)
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("I want to plan a trip to Japan.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Wonderful! When are you thinking?")}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("Sometime in October, flexible dates.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("October is a great time to visit.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("My budget is around $3000.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("That's a reasonable budget.")}),
        ReplayStep(
            op="append_event",
            kwargs={"event": _make_user_event("I want to visit Tokyo and Kyoto mainly.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Both are excellent choices!")}),
        # Create summary (simulates compression, keeps summary + 2 recent events)
        ReplayStep(
            op="create_summary",
            kwargs={
                "summary_text": _SUMMARY_TEXT_8,
                "original_event_count": 8,
                "compressed_event_count": 3,
                "keep_recent_count": 2,
            },
        ),
        # Append new event after truncation
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_user_event("Also, any recommendations for hotels in Tokyo?")
            }),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 9: summary_missing_detection  (corruption case)
# ──────────────────────────────────────────────────────────────
CASE_SUMMARY_MISSING = ReplayCase(
    case_id="summary_missing_detection",
    description="Inject summary loss: one backend has summary, the other does not",
    expected_event_count=4,
    expected_summary_available=True,
    expected_summary_text="User asked about machine learning basics.",
    corruption_type="missing_summary",
    corruption_description="Backend B will have no summary while Backend A does",
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("Tell me about machine learning.")}),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Machine learning is a subset of AI that enables systems to learn from data.")
            }),
        ReplayStep(op="append_event", kwargs={"event": _make_user_event("What about supervised learning?")}),
        ReplayStep(
            op="append_event",
            kwargs={
                "event":
                    _make_agent_text("Supervised learning uses labeled data to train models.")
            }),
        ReplayStep(
            op="create_summary",
            kwargs={
                "summary_text": "User asked about machine learning basics.",
                "original_event_count": 4,
                "compressed_event_count": 3,
            },
        ),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 10: summary_wrong_session  (corruption case)
# ──────────────────────────────────────────────────────────────
CASE_SUMMARY_WRONG_SESSION = ReplayCase(
    case_id="summary_wrong_session",
    description="Inject cross-session summary: summary claims to be for session A but appears in session B",
    expected_event_count=2,
    expected_summary_available=True,
    expected_summary_text="User prefers dark mode and Python development.",
    corruption_type="wrong_session_summary",
    corruption_description="Backend B will have a summary with a different session_id",
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(op="append_event", kwargs={"event": _make_userEvent("I like dark mode.")}),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Noted your dark mode preference.")}),
        ReplayStep(
            op="create_summary",
            kwargs={
                "summary_text": "User prefers dark mode and Python development.",
                "original_event_count": 2,
                "compressed_event_count": 1,
            },
        ),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 11: duplicate_event_detection  (corruption case)
# ──────────────────────────────────────────────────────────────
CASE_DUPLICATE_EVENT = ReplayCase(
    case_id="duplicate_event_detection",
    description="Simulate failed write followed by retry: detect duplicate events",
    expected_event_count=2,
    corruption_type="duplicate_event",
    corruption_description="Backend B will have an extra duplicate event injected",
    steps=[
        ReplayStep(op="create_session", kwargs={"app_name": APP_NAME, "user_id": USER_ID, "session_id": SESSION_ID}),
        ReplayStep(
            op="append_event",
            kwargs={"event": _make_user_event("What is the capital of France?")}),
        ReplayStep(
            op="append_event",
            kwargs={"event": _make_agent_text("The capital of France is Paris.")}),
    ],
)

# ──────────────────────────────────────────────────────────────
# Case 12: state_dirty_after_error  (corruption case)
# ──────────────────────────────────────────────────────────────
CASE_STATE_DIRTY = ReplayCase(
    case_id="state_dirty_after_error",
    description="Simulate partial write failure: one backend has incomplete state",
    expected_event_count=2,
    expected_state={"app:env": "production", "user:role": "admin", "config_key": "expected_value"},
    corruption_type="dirty_state",
    corruption_description="Backend B will have a missing or incorrect state key",
    steps=[
        ReplayStep(
            op="create_session",
            kwargs={
                "app_name": APP_NAME,
                "user_id": USER_ID,
                "session_id": SESSION_ID,
                "state": {"app:env": "staging", "user:role": "viewer"},
            },
        ),
        ReplayStep(
            op="append_event",
            kwargs={
                "event": _make_user_event("Upgrade my account"),
                "state_delta": {
                    "app:env": "production",
                    "user:role": "admin",
                    "config_key": "expected_value",
                },
            },
        ),
        ReplayStep(op="append_event", kwargs={"event": _make_agent_text("Account upgraded successfully.")}),
    ],
)

# ──────────────────────────────────────────────────────────────
# Master list of ALL replay cases
# ──────────────────────────────────────────────────────────────
ALL_REPLAY_CASES: List[ReplayCase] = [
    CASE_SINGLE_TURN,
    CASE_MULTI_TURN,
    CASE_TOOL_CALL,
    CASE_STATE_UPDATE,
    CASE_MEMORY_WRITE_READ,
    CASE_MEMORY_FACTS,
    CASE_SUMMARY_CREATE,
    CASE_SUMMARY_TRUNCATION,
    CASE_SUMMARY_MISSING,
    CASE_SUMMARY_WRONG_SESSION,
    CASE_DUPLICATE_EVENT,
    CASE_STATE_DIRTY,
]
