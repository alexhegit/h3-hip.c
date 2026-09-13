#!/usr/bin/env bash
# Wrapper — canonical script is bench/fox-15s-sched.sh (on-demand, not default scoreboard)
exec "$(dirname "$0")/bench/fox-15s-sched.sh" "$@"
