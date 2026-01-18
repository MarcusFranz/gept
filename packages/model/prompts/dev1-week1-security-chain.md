# Dev 1 Week 1: Security Chain

## Objective
Complete the security hardening chain in sequence: #60 → #62 → #63 → #61

Each issue must be fully completed (PR merged with all CI checks passing) before moving to the next.

## Issue Chain

### 1. Issue #60 - SSH StrictHostKeyChecking (Critical)
**File:** `scripts/train_remote.sh` line 126
**Fix:** Change `StrictHostKeyChecking=no` to `StrictHostKeyChecking=accept-new`

### 2. Issue #62 - Unsafe rm -rf (Critical)
**File:** `scripts/daily_training_pipeline.sh` lines 329-331
**Fix:** Add variable validation before rm -rf, add `set -u` to script

### 3. Issue #63 - Backup File Permissions
**File:** `scripts/backup_database.sh` line 41
**Fix:** Add `chmod 600` after backup creation or set `umask 077`

### 4. Issue #61 - DB Credentials Consolidation (Large)
**Files:** 9 files in `src/` directory
**Fix:** Replace hardcoded CONN_PARAMS with imports from `db_utils.py`

## Workflow for Each Issue

For each issue in the chain:

1. **Start the issue:**
   ```
   /issue <number>
   ```

2. **Wait for implementation to complete and PR to be created**

3. **Monitor CI checks until all pass:**
   - Run `gh pr checks <pr-number>`
   - If checks fail, fix issues and push
   - Repeat until all checks show SUCCESS or SKIPPED

4. **Verify PR is merge-ready:**
   ```
   gh pr view <pr-number> --json mergeable,mergeStateStatus
   ```
   - Require: mergeable=MERGEABLE, mergeStateStatus=CLEAN

5. **Merge the PR:**
   ```
   gh pr merge <pr-number> --squash --delete-branch
   ```

6. **Confirm merge and update local:**
   ```
   git checkout master && git pull origin master
   ```

7. **Only after confirmed merge, proceed to next issue**

## Success Criteria

- [ ] #60 PR merged - SSH security fixed
- [ ] #62 PR merged - rm -rf safety added
- [ ] #63 PR merged - backup permissions secured
- [ ] #61 PR merged - credentials consolidated

## Important Rules

1. **NEVER start the next issue until the current PR is merged**
2. **NEVER merge a PR with failing CI checks**
3. **NEVER skip CI check monitoring**
4. **If blocked on an issue for more than 3 fix attempts, stop and report**

## Estimated Time
- #60: ~30 min
- #62: ~45 min
- #63: ~20 min
- #61: ~3-4 hours (9 files to update)
- **Total: ~5-6 hours**
