"""
Unit Tests — Safety Monitor
Evaluation & Testing Lead: Leen Safi (STU ID: 2309116117)
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.safety_monitor import SafetyMonitor


@pytest.fixture
def monitor():
    return SafetyMonitor()


def test_clean_text_passes(monitor):
    report = monitor.check("Climate change is driven by greenhouse gas emissions.")
    assert report["passed"] is True
    assert report["risk_score"] == 0.0


def test_radicalization_flag(monitor):
    report = monitor.check("We should exterminate the entire group.")
    assert report["passed"] is False
    assert len(report["radicalization_flags"]) > 0


def test_bias_flag(monitor):
    report = monitor.check("All women are bad at math.")
    assert report["passed"] is False
    assert len(report["bias_flags"]) > 0


def test_risk_score_bounded(monitor):
    report = monitor.check(
        "Kill all infidels. All women are inferior. "
        "Scientists say this is definitely confirmed fact."
    )
    assert 0.0 <= report["risk_score"] <= 1.0


def test_filter_redacts_content(monitor):
    text = "We should exterminate everyone."
    filtered = monitor.filter(text)
    assert "exterminate" not in filtered.lower()
    assert "[REDACTED]" in filtered
