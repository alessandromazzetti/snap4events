"""
Shared pytest setup for the whole test/ suite.

Makes the src/ modules (nodes, builder, state, models, db_mcp_server, ...)
importable from every test file here, regardless of which directory pytest
is invoked from (project root, test/, or elsewhere).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(autouse=True)
def _skip_rate_limit_sleeps():
    """extract_events_node() sleeps 2s between scraped URLs to be polite to
    the sites it hits - real and worth keeping in production, but pointless
    to actually wait through in a fast local test suite."""

    with patch("nodes.asyncio.sleep", new=AsyncMock(return_value=None)):
        yield

