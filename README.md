# Music Generation Project

Neural-net based music generation with MIDI input and output

### Installation

Use `pip` or `conda` to install the dependencies:

```bash
pip install -r requirements.txt
```

### Operation

Run the generation server and give it a minute to start:
```bash
python generator_server.py
```

Run the interactive program:
```bash
python piano_player.py
```

### Usage

Play the instrument and the program will read its MIDI output and send notes to its MIDI input.

Change the mode by pressing the corresponding MIDI Program button:

1: Quiet: Generate nothing.

2: Identical: Generate a copy of what was played. Useful for Debugging.

3: Continuation: Generate new notes by using the input as a primer.

4: Accompaniment: Generate a copy of what was played, with accompaniment added.

0: No effect. Press this after selecting a mode to return to the default synthesizer.