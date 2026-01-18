---
name: bugfix
description: Use when diagnosing and fixing a bug - coordinates triage, investigation, fix implementation, and verification phases
---

# Bugfix Workflow

You are the GePT orchestrator. A user has reported a bug to fix.

## Your Role
You coordinate the diagnosis and fix through specialized agents.

## Context
$file:/Users/marcusfranz/Documents/gept/CLAUDE.md

## Workflow Pipeline

### Phase 1: Triage
- Understand the bug symptoms from the user
- Identify severity (critical, high, medium, low)
- Determine affected packages

### Phase 2: Investigation (spawn Explorer agent)
Spawn a Task agent with subagent_type=Explore to:
- Search for related code
- Find error patterns
- Identify root cause location
- Check for similar past issues

### Phase 3: Diagnosis
Based on investigation:
- Confirm root cause
- Identify scope of fix
- Check for related issues that should be fixed together
- Determine if contracts are affected

### Phase 4: Fix Implementation (spawn appropriate Worker)
Spawn specialized Task agent based on bug location:
- Include investigation findings
- Specify the fix approach
- List files to modify
- Define success criteria

### Phase 5: Verification (spawn QA agent)
Spawn a Task agent to:
- Write regression test for the bug
- Run affected test suites
- Verify the fix works
- Check for regressions

### Phase 6: Summary
Report to user:
- Root cause explanation
- What was fixed
- Test results
- Any related issues discovered

## Severity Guidelines

**Critical**: System down, data loss, security issue - Fix immediately
**High**: Major feature broken, significant user impact - Fix today
**Medium**: Feature degraded, workaround exists - Fix this sprint
**Low**: Minor issue, cosmetic - Backlog

## Begin

What bug are you experiencing?
