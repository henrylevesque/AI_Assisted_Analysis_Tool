# Contributing

Thanks for your interest in contributing to the AI Assisted Analysis Tool. This document explains the preferred workflow for reporting issues, proposing features, and submitting code changes.

## Quick checklist

- Open an issue first for anything non-trivial (bug, enhancement, question).
- Use the provided issue templates when creating issues.
- Create a branch named `feat/<short-description>` or `fix/<short-description>` for code changes.
- Open a pull request (PR) against `main` and reference any related issue.
- Ensure your PR includes a short description, tests (if applicable), and updates to docs when behavior changes.

## Development setup (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the project requirements:

```powershell
pip install -r requirements.txt
```

3. Run a quick syntax check locally:

```powershell
python -m compileall .
```

4. Run any module-level scripts as needed:

```powershell
python ai_assisted_analysis.py
```

## Branches & commit messages

- Branches: `feat/`, `fix/`, `docs/`, `chore/`.
- Keep commits small and focused. Use present-tense, imperative messages, e.g. "Add CONTRIBUTING.md".

## Pull request checklist

- [ ] The PR has a clear title and description.
- [ ] The PR references a related issue (if any).
- [ ] Code is formatted and passes a local syntax check.
- [ ] Documentation updated if behavior changed.

## Reporting security issues

If you discover a security vulnerability, please disclose it privately by opening a GitHub issue marked `security` and include steps to reproduce. If you'd prefer, you can email the repository owner listed in `CITATION.cff`.

## Contributor Code of Conduct

Please follow the project's `CODE_OF_CONDUCT.md`.

Thank you for contributing!
