#!/usr/bin/env bash
hatch run format -- --check || {
    echo ""
    echo "Push failed: format violations."
    echo 'Run "hatch run format" to fix.'
    exit 1
}
