# Ralph Loop: Complete Remaining Pre-Release Issues

Copy and paste this command into Claude Code:

```
/ralph-loop:ralph-loop "Complete these 7 GitHub issues for the GePT Recommendation Engine pre-release cleanup.

For each issue:
1. Run /issue <number> to review the issue and implement the fix
2. Create a PR following the worktree workflow in CLAUDE.md
3. Wait for CI tests to pass before merging
4. Only merge the PR after all checks are green
5. Clean up the worktree after merge

Issues to complete in order:
- #89 (security: SQL string interpolation in data_loader.py)
- #94 (bug: predict_all() use_cache parameter is ineffective)
- #88 (refactor: Remove duplicate RecommendationStore class)
- #92 (build: Pin dependency versions in requirements.txt)
- #90 (security: Overly permissive CORS configuration)
- #91 (feature: Add rate limiting to API endpoints)
- #95 (enhancement: Add memory bounds to CrowdingTracker)

Output <promise>ALL_ISSUES_COMPLETE</promise> when all 7 issues have been resolved and merged." --max-iterations 20 --completion-promise "ALL_ISSUES_COMPLETE"
```
