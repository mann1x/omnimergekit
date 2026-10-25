#!/bin/sh
# Don't block headless claude sessions spawned by caliber itself (e.g. during caliber refresh)
if [ "$CALIBER_SUBPROCESS" = "1" ] || [ -n "$CALIBER_SPAWNED" ]; then
  exit 0
fi

# Resolve the project root — the directory that OWNS this .claude/.
#
# Pre-fix (2026-05-19) this script used `git rev-parse --git-dir` against
# $PWD. But the hook command in settings.json is
# "$CLAUDE_PROJECT_DIR/.claude/hooks/caliber-check-sync.sh", and Claude
# Code inherits the SESSION cwd as $PWD — which can be a subdirectory
# of the project that happens to be its own git repo (e.g. a Mythic-RDT/
# checkout nested inside a non-git backup_models/ scratch dir). In that
# layout `git rev-parse` from $PWD succeeded against the nested .git,
# the nudge fired, and the assistant tried to install Caliber in a
# directory where Caliber has nothing to manage.
#
# $CLAUDE_PROJECT_DIR is exported by Claude Code and always points at
# the directory containing the .claude/ that registered this hook —
# that's the only place the Caliber check makes sense. Script-relative
# fallback handles the (rare) shell-test case where the script is
# invoked outside Claude Code (e.g. the Caliber vitest suite).
if [ -n "$CLAUDE_PROJECT_DIR" ]; then
  PROJECT_DIR="$CLAUDE_PROJECT_DIR"
else
  script_dir=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || script_dir=""
  if [ -n "$script_dir" ]; then
    PROJECT_DIR=$(cd "$script_dir/../.." 2>/dev/null && pwd) || PROJECT_DIR="$PWD"
  else
    PROJECT_DIR="$PWD"
  fi
fi

# Caliber only applies to git repos. Skip the nudge when the project
# root has no git context (e.g. .claude/ shipped into a non-git
# scratch dir, model archive, or a parent dir whose only git presence
# is in nested children). `git -C <dir>` confines the lookup to walk
# UP from PROJECT_DIR; it never descends into children.
if ! git -C "$PROJECT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  exit 0
fi
if grep -q "caliber" "$PROJECT_DIR/.git/hooks/pre-commit" 2>/dev/null; then
  exit 0
fi

# Flag is keyed by PROJECT_DIR (not $PWD) so the once-per-project
# guard still works when the assistant moves between subdirectories
# during the session.
FLAG="/tmp/caliber-nudge-$(echo "$PROJECT_DIR" | (shasum 2>/dev/null || sha1sum 2>/dev/null || md5sum 2>/dev/null || cksum) | cut -c1-8)"

# Clean stale flags (older than 2 hours)
find /tmp -maxdepth 1 -name "caliber-nudge-*" -mmin +120 -delete 2>/dev/null

if [ -f "$FLAG" ]; then
  exit 0
fi
touch "$FLAG"

printf '{"decision":"block","reason":"Caliber agent config sync is not set up on this machine. Tell the user: This project uses Caliber to keep agent configs in sync — it is not set up yet. Want me to run /setup-caliber? (~30 seconds)"}'
