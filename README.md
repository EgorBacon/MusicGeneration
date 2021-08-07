# Music Generation Project

Neural-net based music generation with MIDI input and output

### Installation

Use `pip` or `conda` to install the dependencies:

```bash
pip install -r requirements.txt
```

Download the neural net files

```bash
cd content
gsutil -m cp -r 'gs://magentadata/models/music_transformer/checkpoints/*' .
gsutil -q -m cp -r 'gs://magentadata/models/music_transformer/primers/*' .
gsutil -q -m cp 'gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2' .
```

Optional: build the Docker image

```bash
docker build -t music-generation docker
docker run --rm -it --runtime nvidia --device /dev/snd/midiC2D0 --network host -v "$PWD:/app" -w /app music-generation
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

Play the instrument, and the program will read its MIDI output and send notes to its MIDI input.

Change the mode by pressing the corresponding MIDI Program button:

1: Quiet: Generate nothing.

2: Identical: Generate a copy of what was played. Useful for debugging.

3: Continuation: Generate new notes by using the input as a primer.

4: Accompaniment: Generate a copy of what was played, with accompaniment added.

0: No effect. Press this after selecting a mode to return to the default synthesizer.
