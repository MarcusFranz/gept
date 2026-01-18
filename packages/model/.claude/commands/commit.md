---
allowed-tools: Bash(git:*), Read
description: Create a thoughtful commit with a descriptive message
---

# Thoughtful Commit

Create a well-structured git commit with a meaningful message that explains both what changed and why.

## Commit Message Format

Follow this structure:
```
<type>: <short summary in imperative mood>

<body explaining the what and why, not the how>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### Types
- **feat**: New feature or functionality
- **fix**: Bug fix
- **refactor**: Code restructuring without behavior change
- **style**: Formatting, naming, cosmetic changes
- **docs**: Documentation only
- **test**: Adding or updating tests
- **chore**: Maintenance, dependencies, config

### Guidelines
- Subject line: imperative mood, no period, max 50 chars
- Body: wrap at 72 chars, explain context and motivation
- Focus on WHY the change was made, not just WHAT changed
- Reference related issues or decisions if applicable

## Instructions

1. Check git status to see what's staged/unstaged
2. Review the diff to understand all changes
3. Check recent commits for project's style conventions
4. Stage all relevant changes
5. Draft a commit message following the format above
6. Create the commit
7. Optionally push if requested

## Execute

Run these steps:

1. Check status and diff:
```bash
git status
git diff --cached
git diff
```

2. Review recent commit style:
```bash
git log --oneline -5
```

3. Stage changes (if not already staged):
```bash
git add -A
```

4. Analyze the changes thoroughly, then create a commit with a thoughtful message that:
   - Uses a clear, descriptive subject line
   - Includes a body explaining the motivation and impact
   - Groups related changes logically in the description
   - Mentions any breaking changes or important notes

5. Report the commit hash and message to the user
