#!/usr/bin/env bash
hatch run lint || {
    echo ""
    echo "Push failed: ruff lint errors."
    echo 'Run "hatch run lint -- --fix" to auto-fix.'
    exit 1
}
