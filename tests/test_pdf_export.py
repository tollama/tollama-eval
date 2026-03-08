"""Tests for PDF export module."""

from __future__ import annotations

from ts_autopilot.reporting.pdf_export import is_available


def test_is_available_returns_bool():
    result = is_available()
    assert isinstance(result, bool)
