from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


BLOCKED_EXTENSIONS = {".docx", ".pdf", ".zip", ".pem", ".key", ".p12", ".pfx"}
SKIPPED_DIRS = {".git", ".venv", "__pycache__"}
PRIVATE_POSITIONING_TERMS = [
    "F" + "DE",
    "Forward" + " " + "Deployed",
    "inter" + "view",
    "job" + "-" + "application",
    "rec" + "ruiter",
    "target" + "-" + "role",
    "hir" + "ing",
    "Google" + " " + "roles",
]
SECRET_PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("google api key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("oauth access token", re.compile(r"ya29\.[0-9A-Za-z_-]+")),
    ("github token", re.compile(r"(ghp|github_pat)_[0-9A-Za-z_]+")),
    ("slack token", re.compile(r"xox[baprs]-[0-9A-Za-z-]+")),
    ("service account private key", re.compile(r'"private_key"\s*:\s*"')),
    ("service account client email", re.compile(r'"client_email"\s*:\s*"[^"]+@[^"]+\.iam\.gserviceaccount\.com"')),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("local ssh key path", re.compile(r"/\.ssh/|sshkey_[A-Za-z0-9_-]+")),
    ("concrete gcp project id", re.compile(r"\bproject-[0-9a-f]{8}-[0-9a-f-]{27,}\b")),
    ("vertex resource id", re.compile(r"projects/[^/\s]+/locations/[^/\s]+/(indexes|indexEndpoints)/[^/\s]+")),
]


@dataclass(frozen=True)
class Finding:
    path: str
    line_number: int
    kind: str
    line: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Check tracked public files for private artifacts and secrets.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    tracked = tracked_files(root)
    findings = check_files(root, tracked)
    if findings:
        for finding in findings:
            print(f"{finding.path}:{finding.line_number}: {finding.kind}: {finding.line}")
        raise SystemExit(1)
    print(f"public hygiene passed: {len(tracked)} tracked files checked")


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def check_files(root: Path, paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_extension_findings(path))
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        full_path = root / path
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(_content_findings(path, text))
    return findings


def _extension_findings(path: Path) -> list[Finding]:
    if path.suffix.lower() in BLOCKED_EXTENSIONS:
        return [Finding(str(path), 0, "blocked tracked file type", "downloaded documents and key files must not be tracked")]
    return []


def _content_findings(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for term in PRIVATE_POSITIONING_TERMS:
            if term in line:
                findings.append(Finding(str(path), line_number, "private positioning language", line.strip()))
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(str(path), line_number, name, _redact(line.strip())))
    return findings


def _redact(line: str) -> str:
    if len(line) <= 160:
        return line
    return line[:157] + "..."


if __name__ == "__main__":
    main()
