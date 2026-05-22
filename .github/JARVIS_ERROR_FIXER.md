# Jarvis Error Fixer Agent

A specialized VS Code Copilot agent designed to diagnose and fix errors in the Jarvis multi-mode AI assistant project.

## Overview

The **Jarvis Error Fixer** is a custom agent that analyzes the Jarvis codebase for:

- **Import errors** - Missing or incorrect module imports
- **Dependency issues** - Mismatched or missing packages in `pyproject.toml`
- **Syntax errors** - Python code compilation issues
- **Runtime failures** - Issues that appear when code executes
- **Configuration problems** - Invalid YAML configs or missing environment variables
- **Platform compatibility** - Cross-platform issues (Windows/macOS/Linux)

## How to Use

### Option 1: Direct Agent Invocation (In VS Code)

In the Copilot chat, invoke the agent with:

```
@jarvis-fixer fix run_jarvis.py
@jarvis-fixer diagnose imports
@jarvis-fixer validate project
```

The agent runs in its specialized context with tools restricted to project analysis and fixing.

### Option 2: Use the Fix-Jarvis-Errors Prompt

In the Copilot chat, use the built-in prompt:

```
/fix-jarvis-errors all
/fix-jarvis-errors imports
/fix-jarvis-errors dependencies
/fix-jarvis-errors runtime
```

### Option 3: Manual Invocation with Parameters

```
Analyze Jarvis project scope:imports
Check Jarvis for syntax errors
Find all import failures in Jarvis
```

## What Gets Analyzed

### Core Files
- `run_jarvis.py` - Main orchestrator (terminal, email, HF modes)
- `email_handler.py` - IMAP/SMTP polling
- `system_monitor.py` - System vitals monitoring
- `src/my_log_agent/crew.py` - CrewAI crew definition
- `src/my_log_agent/main.py` - CrewAI entrypoint

### Configuration Files
- `pyproject.toml` - Dependencies and project metadata
- `src/my_log_agent/config/agents.yaml` - Agent definitions (if exists)
- `src/my_log_agent/config/tasks.yaml` - Task definitions (if exists)

### Test/Diagnostic Scripts
- `hf_test.py` - Hugging Face integration test
- `test_email.py` - Email system test
- `hf_test_models.py` - HF model availability test

## Known Issues Fixed By This Agent

### Issue: `ModuleNotFoundError: No module named 'yaml'`
- **Problem**: `pyyaml` not in dependencies
- **Fix**: Add to `pyproject.toml` dependencies
- **Validation**: `python3 -c "import yaml; print(yaml.__version__)"`

### Issue: CrewAI API Incompatibility
- **Problem**: Using outdated CrewAI API (e.g., `ChatOpenAI()` instead of `LLM()`)
- **Fix**: Update agent/crew code to match installed CrewAI version
- **Validation**: Check `crewai.__version__` and compare against AGENTS.md reference

### Issue: Platform-Specific UTF-8 Encoding
- **Problem**: Email parsing or file I/O fails on Windows
- **Fix**: Add explicit UTF-8 encoding specifications
- **Validation**: Test on Windows or mock UTF-8 scenarios

### Issue: Missing Environment Variable Defaults
- **Problem**: Code crashes when `.env` key is missing
- **Fix**: Add `os.getenv("KEY", "default_value")` patterns
- **Validation**: Run without `.env` set

## Output

When the agent completes analysis, it provides:

1. **Error Summary** - Total errors found, categorized by type
2. **Detailed Findings** - Each error with location, cause, and fix
3. **Applied Fixes** - List of changes made to source code
4. **Validation Report** - Test results confirming fixes work
5. **Next Steps** - Remaining manual work (e.g., credentials, external services)

## Restrictions

The agent is restricted from:
- Modifying `.git` history
- Reading/writing `.env` credentials
- Deleting project files (only edits/adds)
- Executing arbitrary scripts
- Accessing files outside the Jarvis project directory

## Customization

To add more analysis or custom fix logic, edit:
- `.github/agents/jarvis-fixer.agent.md` - Agent definition and instructions
- `.github/prompts/fix-jarvis-errors.prompt.md` - Prompt parameters and flow

## Related Files

```
.github/
├── agents/
│   └── jarvis-fixer.agent.md       # Agent definition
└── prompts/
    └── fix-jarvis-errors.prompt.md # Prompt template
```

## Troubleshooting

**Agent doesn't appear in chat?**
- Ensure `.github/agents/jarvis-fixer.agent.md` exists and has valid YAML frontmatter
- Reload VS Code window (Cmd+R / Ctrl+R)

**Agent can't edit files?**
- Verify the file is not read-only: `chmod +w <file>`
- Check that file path is within the Jarvis project directory

**Fixes don't work?**
- Agent may have identified the issue but need manual intervention (e.g., installing system packages)
- Check agent output for "manual intervention required" sections
- Run validation commands from the report manually

## See Also

- **AGENTS.md** - CrewAI API reference for agent patterns
- **README.md** - Jarvis project overview and setup
- **pyproject.toml** - Project dependencies
