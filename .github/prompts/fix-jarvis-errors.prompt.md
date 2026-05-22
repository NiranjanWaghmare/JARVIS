---
name: fix-jarvis-errors
description: "Diagnose and fix errors in the Jarvis project. Analyzes imports, dependencies, syntax, and CrewAI configuration. Returns a detailed error report with fixes applied and validation results."
model: claude-sonnet-4-20250514
tools:
  - bash
  - view
  - edit
  - grep
  - glob
cwd: /Users/niranjanwaghmare/Desktop/Projects/Jarvis
parameters:
  - name: scope
    description: "What to analyze: 'all' (full project), 'imports' (import errors), 'dependencies' (missing packages), 'syntax' (code compilation), 'runtime' (try running), or a specific file"
    type: string
    default: "all"
---

# Fix Jarvis Errors

You are diagnosing and fixing errors in the **Jarvis project** — a multi-mode CrewAI-based assistant.

## Task

Analyze the Jarvis project with scope: **{{scope}}**

### Analysis Steps

1. **Check structure**: List all Python files and config files
2. **Check dependencies**: Review pyproject.toml and verify imports
3. **Compile check**: Run `python3 -m py_compile` on all .py files
4. **Import check**: Try to import key modules and catch errors
5. **Config validation**: Check if YAML configs exist and are valid
6. **Runtime check** (if scope includes): Attempt `python3 run_jarvis.py 2>&1` and capture errors

### Reporting

For each error found:
- **Location**: File name and line number
- **Error type**: Import error, syntax error, dependency missing, etc.
- **Root cause**: Why it's failing
- **Fix**: Proposed solution with code
- **Validation**: How to verify the fix works

### When Analysis is Complete

Summarize:
- Total errors found
- Errors fixed (with list)
- Errors requiring manual intervention (if any)
- Recommended next steps
