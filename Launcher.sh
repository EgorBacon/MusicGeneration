#!/usr/bin/sh

cd "$(dirname "$0")"

conda activate magenta

python3 GeneratorServer.py &
python3 piano_player.py

