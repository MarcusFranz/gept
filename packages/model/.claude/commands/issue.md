---
allowed-tools: Bash(gh:*), Bash(git:*), Bash(cd:*), Bash(pwd), Bash(pytest:*), Bash(npm:*), Bash(pip:*), Bash(flake8:*), Bash(black:*), Bash(eslint:*), Bash(sleep:*), Read, Grep, Glob, Task, Edit, Write, TodoWrite, EnterPlanMode
description: Review a GitHub issue and formulate a plan to create a PR
argument-hint: <issue number>
---

# GitHub Issue to PR Workflow (Parallel-Safe)

Review a GitHub issue, create an isolated worktree, implement changes, create a PR, monitor CI checks, auto-fix issues, merge the PR, and report completion.

**IMPORTANT**: This workflow uses git worktrees to support multiple parallel Claude instances working on different issues simultaneously.

## Workflow Overview

1. **Check Issue Status** - Verify issue is actionable (not backlogged/needs-info)
2. **Setup Worktree** - Create isolated working directory
3. **Fetch Issue** - Get full issue details
4. **Analyze & Plan** - Explore codebase and create implementation plan (or request more info)
5. **Implement** - Make changes in the isolated worktree
6. **Create PR** - Push and open PR referencing the issue
7. **Monitor CI** - Wait for lint and code review checks, auto-fix failures
8. **Merge & Cleanup** - Merge PR, remove worktree, report completion

---

## Phase 0: Check Issue Status (MUST DO FIRST)

### Step 0.1: Fetch Issue Labels

Before doing any work, check if the issue is actionable:

\`\`\`bash
gh issue view $ARGUMENTS --json labels -q '.labels[].name'
\`\`\`

### Step 0.2: Check for Blocking Labels

If the issue has **"backlog"** or **"needs-info"** labels:

**STOP IMMEDIATELY** and output:

\`\`\`
============================================
Issue Backlogged/Needs Info - Ready for Next Issue
============================================

Issue: #$ARGUMENTS
Status: Not actionable

Reason: Issue has "<label>" tag
Action: Skipping this issue

Ready for next issue.
============================================
\`\`\`

**Do NOT proceed with any further steps. Do NOT create a worktree.**

---

## Phase 1: Setup Isolated Worktree

### Step 1.1: Determine Repository Root

First, identify the repository root and get repo info:

\`\`\`bash
# Get the repo root
REPO_ROOT=\$(git rev-parse --show-toplevel)
REPO_NAME=\$(basename "\$REPO_ROOT")
REPO_PARENT=\$(dirname "\$REPO_ROOT")
echo "Repo: \$REPO_NAME at \$REPO_ROOT"
pwd
\`\`\`

### Step 1.2: Determine Default Branch

\`\`\`bash
# Get default branch (main or master)
DEFAULT_BRANCH=\$(git remote show origin | grep 'HEAD branch' | cut -d' ' -f5)
echo "Default branch: \$DEFAULT_BRANCH"
\`\`\`

### Step 1.3: Create Worktree

Create an isolated worktree for this issue in the parent directory:

\`\`\`bash
# Fetch latest from origin
git fetch origin

# Create worktree with a branch for this issue
# Branch naming: feat/issue-XX or fix/issue-XX (refined after reading issue)
git worktree add "\$REPO_PARENT/\${REPO_NAME}-issue-\$ARGUMENTS" -b feat/issue-\$ARGUMENTS "origin/\$DEFAULT_BRANCH"
\`\`\`

### Step 1.4: Navigate to Worktree

**CRITICAL**: All subsequent work MUST happen in the worktree directory:

\`\`\`bash
cd "\$REPO_PARENT/\${REPO_NAME}-issue-\$ARGUMENTS"
pwd
\`\`\`

**Verify you are in the correct worktree before proceeding.**

---

## Phase 2: Fetch and Analyze Issue

### Step 2.1: Fetch Issue Details

\`\`\`bash
gh issue view \$ARGUMENTS --json title,body,labels,comments,state,author
\`\`\`

### Step 2.2: Analyze Requirements

After fetching the issue:

1. Parse the issue requirements carefully
2. Identify the type of change:
   - \`feat/\` for features
   - \`fix/\` for bug fixes
   - \`docs/\` for documentation
   - \`refactor/\` for restructuring
   - \`test/\` for test additions
3. If branch type should change from \`feat/\`, rename it:
   \`\`\`bash
   git branch -m feat/issue-\$ARGUMENTS <correct-prefix>/issue-\$ARGUMENTS-description
   \`\`\`

### Step 2.3: Check if More Information is Needed

**CRITICAL**: If the issue is unclear, ambiguous, or missing critical information needed to implement:

1. **Leave a comment on the issue** explaining what information is needed:
   \`\`\`bash
   gh issue comment \$ARGUMENTS --body "## Clarification Needed

   To implement this issue, I need the following information:

   - <specific question 1>
   - <specific question 2>
   - <etc>

   Please provide these details so I can proceed with implementation.

   ---
   *Automated comment from Claude Code*"
   \`\`\`

2. **Add the "needs-info" label**:
   \`\`\`bash
   gh issue edit \$ARGUMENTS --add-label "needs-info"
   \`\`\`

3. **Clean up the worktree** (since we're stopping):
   \`\`\`bash
   cd "\$REPO_ROOT"
   git worktree remove "\$REPO_PARENT/\${REPO_NAME}-issue-\$ARGUMENTS"
   git branch -D feat/issue-\$ARGUMENTS
   \`\`\`

4. **STOP and output**:
   \`\`\`
   ============================================
   Issue Needs More Information - Ready for Next Issue
   ============================================

   Issue: #\$ARGUMENTS
   Status: Needs clarification

   Comment left on issue requesting:
   - <list what was asked>

   Label "needs-info" added.

   Ready for next issue.
   ============================================
   \`\`\`

**Do NOT proceed if information is missing.**

### Step 2.4: Explore Codebase

Use the Task tool with Explore agent to find relevant code and understand the implementation context.

### Step 2.5: Create Implementation Plan

1. Enter plan mode using \`EnterPlanMode\` tool
2. Write a detailed implementation plan
3. Exit plan mode and wait for user approval

**Do NOT proceed to implementation until user approves the plan.**

---

## Phase 3: Implementation

### Step 3.1: Make Code Changes

Working in the worktree directory:

1. Make the necessary code changes
2. Follow existing code patterns and style
3. Keep changes focused and atomic

### Step 3.2: Run Local Validation

Before committing, run local checks appropriate for the project:

**For Python projects:**
\`\`\`bash
pip install flake8 black pytest pytest-cov
flake8 src/ --max-line-length=120 --ignore=E501,W503
black --check src/
pytest tests/ -v
\`\`\`

**For Node.js projects:**
\`\`\`bash
npm install
npm run lint
npm test
\`\`\`

**Fix any local failures before committing.**

### Step 3.3: Commit Changes

\`\`\`bash
git add -A
git status

# Commit with descriptive message referencing the issue
git commit -m "<type>: <description>

<body explaining the changes>

Closes #\$ARGUMENTS

Co-Authored-By: Claude <noreply@anthropic.com>"
\`\`\`

---

## Phase 4: Create Pull Request

### Step 4.1: Push Branch

\`\`\`bash
git push -u origin HEAD
\`\`\`

### Step 4.2: Create PR

\`\`\`bash
gh pr create --title "<type>: <description>" --body "## Summary
<1-3 bullet points describing the changes>

## Test Plan
- [ ] Local tests pass
- [ ] Linting passes
- [ ] Changes verified manually

Closes #\$ARGUMENTS

---
Generated with Claude Code"
\`\`\`

### Step 4.3: Get PR Number

\`\`\`bash
# Store PR number for check monitoring
gh pr view --json number -q '.number'
\`\`\`

---

## Phase 5: Monitor CI Checks and Auto-Fix (MAX 5 ITERATIONS)

This is the critical phase that ensures PR quality before merging.

**IMPORTANT: Maximum 5 fix iterations allowed. Track your iteration count.**

### Step 5.1: Initialize Iteration Counter

Start with iteration = 1. After each fix attempt, increment the counter.

### Step 5.2: Wait for Checks to Start

\`\`\`bash
# Wait for checks to register (usually takes 10-30 seconds)
sleep 15
\`\`\`

### Step 5.3: Monitor Check Status Loop

Repeatedly check CI status until all checks pass or need intervention:

\`\`\`bash
# Get current check status
gh pr checks --json name,state,conclusion
\`\`\`

**Check Status Values:**
- \`PENDING\` / \`IN_PROGRESS\` - Still running, wait and check again
- \`SUCCESS\` - Check passed
- \`FAILURE\` - Check failed, needs fixing

### Step 5.4: Handle Check Failures

**Before attempting any fix, check iteration count. If iteration > 5, go to Step 5.5.**

If any check fails and iteration <= 5:

1. **Get failure details:**
   \`\`\`bash
   # View the failed check run logs
   gh run list --branch \$(git branch --show-current) --json name,status,conclusion,databaseId -q '.[] | select(.conclusion == "failure")'

   # Get detailed logs for failed run
   gh run view <run-id> --log-failed
   \`\`\`

2. **Analyze the failure:**
   - Linting error: Fix formatting/style issues
   - Test failure: Fix the failing test or implementation bug
   - Code review feedback: Address the reviewer's concerns

3. **Fix the issues:**
   - Make necessary code changes
   - Re-run local validation (Step 3.2)
   - Commit the fix:
     \`\`\`bash
     git add -A
     git commit -m "fix: address CI feedback (attempt <iteration>/5)

     - <describe fixes made>

     Co-Authored-By: Claude <noreply@anthropic.com>"
     \`\`\`

4. **Push the fix:**
   \`\`\`bash
   git push
   \`\`\`

5. **Increment iteration counter** (iteration = iteration + 1)

6. **Return to Step 5.2** - Wait for new checks to run

### Step 5.5: Iteration Limit Reached (STOP)

If iteration > 5 and checks are still failing:

1. **Do NOT attempt any more fixes**

2. **Leave a comment on the issue explaining the problem:**
   \`\`\`bash
   gh issue comment \$ARGUMENTS --body "## CI/Review Failed - Max Iterations Reached

   I attempted to fix CI/review failures 5 times but was unable to resolve all issues.

   ### Failing Checks
   - <list failing checks>

   ### Recurring Errors
   - <summarize the errors that kept occurring>

   ### What Was Tried
   - <brief summary of fix attempts>

   ### Recommended Next Steps
   - <specific guidance on what might fix the issue>

   The PR #<pr-number> remains open for manual intervention.

   ---
   *Automated comment from Claude Code*"
   \`\`\`

3. **Add the "needs-info" label to the issue:**
   \`\`\`bash
   gh issue edit \$ARGUMENTS --add-label "needs-info"
   \`\`\`

4. **Clean up the worktree:**
   \`\`\`bash
   cd "\$REPO_ROOT"
   git worktree remove "\$REPO_PARENT/\${REPO_NAME}-issue-\$ARGUMENTS"
   \`\`\`

5. **Output and STOP:**
   \`\`\`
   ============================================
   CI/Review Failed - Max Iterations Reached
   ============================================

   Issue: #\$ARGUMENTS
   PR: #<pr-number>
   Status: FAILED after 5 fix attempts

   Failing checks:
   - <list failing checks>

   Last errors:
   - <summarize recurring issues>

   Comment left on issue explaining failures.
   Label "needs-info" added.
   PR remains open for manual intervention.
   Worktree cleaned up.

   Ready for next issue.
   ============================================
   \`\`\`

**Do NOT continue to Phase 6 if iteration limit is reached.**

---

## Phase 6: Verify Merge-Ready Status

### Step 6.1: Check for Merge Conflicts

\`\`\`bash
# Fetch latest default branch
git fetch origin \$DEFAULT_BRANCH

# Check if branch can be merged cleanly
git merge-tree \$(git merge-base HEAD origin/\$DEFAULT_BRANCH) HEAD origin/\$DEFAULT_BRANCH
\`\`\`

If conflicts exist:

\`\`\`bash
# Rebase onto latest default branch
git rebase origin/\$DEFAULT_BRANCH

# If rebase has conflicts, resolve them:
# 1. Fix conflicting files
# 2. git add <resolved-files>
# 3. git rebase --continue

# Force push the rebased branch
git push --force-with-lease
\`\`\`

Then return to Phase 5 to verify checks still pass after rebase.

### Step 6.2: Final Check Verification

\`\`\`bash
# Verify ALL checks are passing
gh pr checks

# Verify PR is mergeable
gh pr view --json mergeable,mergeStateStatus -q '{mergeable: .mergeable, status: .mergeStateStatus}'
\`\`\`

**Required for merge:**
- All checks show \`SUCCESS\` or \`SKIPPED\`
- \`mergeable\` is \`MERGEABLE\`
- \`mergeStateStatus\` is \`CLEAN\`

---

## Phase 7: Merge PR and Cleanup

### Step 7.1: Merge the PR

Once all checks pass and PR is mergeable:

\`\`\`bash
# Merge the PR using squash merge
gh pr merge --squash --delete-branch
\`\`\`

### Step 7.2: Cleanup Worktree

\`\`\`bash
# Return to main repo
cd "\$REPO_ROOT"

# Remove the worktree
git worktree remove "\$REPO_PARENT/\${REPO_NAME}-issue-\$ARGUMENTS"

# Fetch to update local refs
git fetch origin
\`\`\`

### Step 7.3: Final Success Output

\`\`\`
============================================
Issue Complete - Ready for Next Issue
============================================

Issue: #\$ARGUMENTS
PR: #<pr-number>
Status: MERGED

All CI checks passed:
- CI (lint, tests): SUCCESS
- Claude Code Review: SUCCESS

PR merged and branch deleted.
Worktree cleaned up.

Ready for next issue.
============================================
\`\`\`

---

## Blocked Case Output

If the PR cannot be merged after all attempts:

\`\`\`
============================================
PR NOT Ready - Manual Intervention Required
============================================

PR: #<pr-number>
Branch: <branch-name>

Blockers:
- <list specific issues>

Recommended actions:
- <specific guidance>

NOT ready for next issue - requires manual fix.
============================================
\`\`\`

---

## Important Notes

1. **Check labels first** - Always verify issue is actionable before creating worktree
2. **Request info early** - If requirements are unclear, ask immediately and move on
3. **Always work in the worktree** - Never modify files in the main repository directory
4. **Local validation first** - Always run linters and tests locally before pushing
5. **Incremental fixes** - Fix one issue at a time and push to verify
6. **Patience with CI** - Some checks may take 2-5 minutes to complete
7. **MAX 5 ITERATIONS** - Stop after 5 CI/review fix attempts, leave PR open for manual intervention
8. **User approval** - Always wait for plan approval before implementing
9. **Merge when ready** - Don't leave PRs hanging, merge them when CI passes
10. **Clean up always** - Remove worktrees after completion or when stopping early
