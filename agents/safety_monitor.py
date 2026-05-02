"""
Safety Monitor — Guardrails for Hallucinations, Bias, and Radicalization Risks
Risk Management & Safety Monitor: Abdulaziz ALYAHYA (STU ID: 2309116441)
"""

import re
from dataclasses import dataclass, field


RADICALIZATION_KEYWORDS: list[str] = [
    "jihad", "infidel", "kill all", "how to kill", "how to murder",
    "how to harm", "how to hurt", "exterminate", "white supremacy",
    "ethnic cleansing", "race war", "genocide", "terrorize",
    "how to make a bomb", "how to make a weapon", "how to poison",
    "how to attack", "how to shoot",
]

BIAS_INDICATORS: list[str] = [
    "all women are", "all men are", "black people always", "white people always",
    "muslims always", "christians always", "jews always", "asians always",
]

HALLUCINATION_PATTERNS: list[str] = [
    r"\b(definitely|certainly|absolutely|always|never)\b.*\b(proven|confirmed|fact)\b",
    r"\bstudy shows\b(?!.*\bcitation\b)",
    r"\bscientists say\b(?!.*\baccording to\b)",
]


@dataclass
class SafetyReport:
    passed: bool
    radicalization_flags: list[str] = field(default_factory=list)
    bias_flags: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    risk_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "risk_score": round(self.risk_score, 3),
            "radicalization_flags": self.radicalization_flags,
            "bias_flags": self.bias_flags,
            "hallucination_flags": self.hallucination_flags,
        }


class SafetyMonitor:
    """
    Runs three layers of safety checks on every agent output:
      1. Radicalization keyword detection
      2. Implicit bias indicator scanning
      3. Hallucination pattern heuristics
    """

    def check(self, text: str) -> dict:
        report = SafetyReport(passed=True)
        lower = text.lower()

        for kw in RADICALIZATION_KEYWORDS:
            if kw in lower:
                report.radicalization_flags.append(kw)

        for indicator in BIAS_INDICATORS:
            if indicator in lower:
                report.bias_flags.append(indicator)

        for pattern in HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                report.hallucination_flags.extend([str(m) for m in matches])

        total_flags = (
            len(report.radicalization_flags)
            + len(report.bias_flags)
            + len(report.hallucination_flags)
        )
        report.risk_score = min(1.0, total_flags * 0.15)
        report.passed = total_flags == 0

        return report.to_dict()

    def filter(self, text: str) -> str:
        """Return sanitized text — replace flagged phrases with [REDACTED]."""
        lower = text.lower()
        for kw in RADICALIZATION_KEYWORDS + BIAS_INDICATORS:
            if kw in lower:
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                text = pattern.sub("[REDACTED]", text)
        return text
