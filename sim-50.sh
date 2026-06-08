#!/bin/bash
# 50-game driver — 50 games per opponent class, 30 / 10 / 10 role split
# (60 % liberal / 20 % fascist / 20 % hitler — matches the 60/20/20 schedule
# in the 100-game sim.sh, just scaled down).
#
# Usage: bash sim-50.sh <config.yaml>

CONFIG_FILE=${1:-config-local.yaml}

source .venv/bin/activate

MAX_PARALLEL_GAMES=${MAX_PARALLEL_GAMES:-4}

wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL_GAMES" ]; do
        sleep 5
    done
}

start_time=$(date)
echo "Running 50 games with up to $MAX_PARALLEL_GAMES parallel processes at $(date)..."
echo "Using config: $CONFIG_FILE"
echo "Alice (Player 0) role distribution: 30x Liberal, 10x Fascist, 10x Hitler"

for i in $(seq 1 50); do
    wait_for_slot

    # Games 1-30: Liberal, 31-40: Fascist, 41-50: Hitler
    if [ "$i" -le 30 ]; then
        ROLE="liberal"
    elif [ "$i" -le 40 ]; then
        ROLE="fascist"
    else
        ROLE="hitler"
    fi

    echo "Run $i (Alice: $ROLE)"
    (python simulator/HitlerGame.py --config "$CONFIG_FILE" --role "$ROLE" \
        || echo "Run $i failed, continuing with next run") &
done

echo "Waiting for all games to complete..."
wait
echo "All games completed at $(date)!"
