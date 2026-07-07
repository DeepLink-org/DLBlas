"""Append LLM follow-up diagnosis content to generated reports."""

from __future__ import annotations

import re
from pathlib import Path

SECTION_TITLE = "## 9. LLM Follow-up Diagnosis"
START_MARKER = "<!-- trace-report:llm-follow-up-diagnosis:start -->"
END_MARKER = "<!-- trace-report:llm-follow-up-diagnosis:end -->"
REQUIRED_HEADINGS = (
    "Summary",
    "Quality Gate Result",
    "Final Diagnosis",
    "Validation Plan",
)
REASONING_CHAIN_REQUIREMENTS = (
    ("measured evidence", ("measured evidence", "measured:")),
    ("mechanism", ("mechanism",)),
    ("counter-evidence / weaker hypotheses", ("counter-evidence", "weaker hypotheses")),
    ("confidence boundary", ("confidence boundary",)),
    ("edit rationale", ("edit rationale",)),
    ("validation expectation", ("validation expectation",)),
)


def _section(content: str) -> str:
    body = content.strip()
    return f"{SECTION_TITLE}\n\n{START_MARKER}\n{body}\n{END_MARKER}\n"


def _has_heading(content: str, heading: str) -> bool:
    marker = f"### {heading}"
    return any(line.strip() == marker for line in content.splitlines())


def _top_level_section(content: str, heading: str) -> str:
    pattern = re.compile(rf"^### {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(content)
    if match is None:
        return ""
    next_match = re.search(r"^### (?!#).+", content[match.end():], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(content)
    return content[match.end():end]


def _validate_reasoning_chain(diagnosis: str, index: int) -> list[str]:
    match = re.search(r"^#### Reasoning Chain\s*$", diagnosis, re.MULTILINE)
    if match is None:
        return [f"Diagnosis {index} is missing #### Reasoning Chain"]
    chain = diagnosis[match.end():].lower()
    missing = [
        label
        for label, aliases in REASONING_CHAIN_REQUIREMENTS
        if not any(alias in chain for alias in aliases)
    ]
    if missing:
        return [
            f"Diagnosis {index} Reasoning Chain is missing: " + ", ".join(missing)
        ]
    return []


def validate_per_diagnosis_reasoning(content: str) -> None:
    final_diagnosis = _top_level_section(content, "Final Diagnosis")
    diagnosis_matches = list(re.finditer(r"^#### Diagnosis\b.*$", final_diagnosis, re.MULTILINE))
    if not diagnosis_matches:
        raise ValueError("Final Diagnosis must include at least one #### Diagnosis item")

    errors: list[str] = []
    for idx, match in enumerate(diagnosis_matches, 1):
        next_match = diagnosis_matches[idx] if idx < len(diagnosis_matches) else None
        end = next_match.start() if next_match else len(final_diagnosis)
        errors.extend(_validate_reasoning_chain(final_diagnosis[match.end():end], idx))
    if errors:
        raise ValueError("; ".join(errors))


def validate_llm_followup_diagnosis(content: str) -> None:
    missing = [heading for heading in REQUIRED_HEADINGS if not _has_heading(content, heading)]
    if missing:
        raise ValueError("LLM follow-up diagnosis is missing required headings: " + ", ".join(missing))
    validate_per_diagnosis_reasoning(content)


def append_llm_followup_diagnosis(report_path: Path, content: str) -> None:
    """Append or replace the LLM follow-up diagnosis section in a report."""
    body = content.strip()
    if not body:
        raise ValueError("LLM follow-up diagnosis content is empty")
    validate_llm_followup_diagnosis(body)
    if not report_path.exists():
        raise FileNotFoundError(f"report does not exist: {report_path}")

    report = report_path.read_text(encoding="utf-8")
    start = report.find(START_MARKER)
    end = report.find(END_MARKER)
    if (start == -1) != (end == -1):
        raise ValueError("report has incomplete LLM follow-up diagnosis markers")

    section = _section(body)
    if start != -1:
        end += len(END_MARKER)
        heading = report.rfind(SECTION_TITLE, 0, start)
        replace_start = heading if heading != -1 else start
        updated = report[:replace_start].rstrip() + "\n\n" + section + report[end:].lstrip("\n")
    else:
        updated = report.rstrip() + "\n\n" + section

    report_path.write_text(updated, encoding="utf-8")
