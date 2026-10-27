"""Tests run fully offline: rules-only detection, MockLaya decisions, MockLLM, throwaway SQLite DB.
(Tests that need the real GLiNER / Laya weights are marked `ml` and skipped unless LG_TEST_ML=1.)"""

import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="leakguard-test-")
os.environ.setdefault("LG_DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ.setdefault("LG_USE_GLINER", "false")
os.environ.setdefault("LG_USE_LAYA", "false")
os.environ.setdefault("LG_LLM_PROVIDER", "mock")
os.environ.setdefault("LG_VAULT_KEY", "Qx3P1Z3l4kq1q6eB5gC6H8bJ1oM3Yk3cG9a0nW2uJ4E=")

from leakguard.storage.db import init_db  # noqa: E402

init_db()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LG_TEST_ML") == "1":
        return
    skip = pytest.mark.skip(reason="needs model weights; set LG_TEST_ML=1")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def guard():
    from leakguard.agent.graph import run_guard

    return run_guard


@pytest.fixture(autouse=True)
def _isolated_memory():
    """Reviewer precedents persist by design (feedback loop); isolate them per test."""
    from leakguard.agent.memory import precedents
    from leakguard.storage.db import session_scope
    from leakguard.storage.models import MemoryItem

    with session_scope() as s:
        s.query(MemoryItem).delete()
    precedents.invalidate()
    yield
