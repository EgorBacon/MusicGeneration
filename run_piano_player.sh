#!/bin/sh -eu

cd "$(dirname "$0")"

conda activate magenta

python3 generator_server.py &

python3 piano_player.py
