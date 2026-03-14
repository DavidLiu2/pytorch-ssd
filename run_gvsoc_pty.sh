#!/usr/bin/env bash
set -euo pipefail
docker exec aideck bash -lc 'cd /module/aideck-gap8-examples/examples/other/dory_examples/application && rm -f run_tty.log && script -q -c "bash -lc '\''cd /gap_sdk && source configs/ai_deck.sh && cd /module/aideck-gap8-examples/examples/other/dory_examples/application && make run platform=gvsoc'\''" run_tty.log'