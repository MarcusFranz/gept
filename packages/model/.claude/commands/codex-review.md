---
allowed-tools: Bash(git:*), Bash(codex:*)
description: Send staged files to Codex for code review before committing
---

# Codex Review

Review uncommitted changes with Codex before committing.

## Instructions

1. Stage all changes with `git add -A`
2. Run `codex review --uncommitted` to review all staged/unstaged changes
3. If Codex identifies issues and provides fix commands, execute them
4. Re-stage any modified files after fixes are applied
5. Report findings to the user

## Execute

Run the following steps:

1. Stage all changes:
```bash
git add -A
```

2. Run Codex review on uncommitted changes:
```bash
codex review --uncommitted
```

3. If Codex returns fix commands, execute them

4. After fixes are applied, re-stage modified files:
```bash
git add -A
```

5. Report what was found/fixed to the user
