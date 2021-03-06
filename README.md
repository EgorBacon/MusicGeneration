# Piano Player
This is a ML project which is designed to tak input from a MIDI keyboard and then play along in four modes: Replay, Quiet, Improv and Accompaniment. Although this is still under development and has a few issues.

# Equipment:
For this project you need three things:
* computer that can run tensorflow
* MIDI keyboard that can be either USB powered or connected over bluetooth (see Branch "Jesse" for bluetooth connectivity)

# Installation:
1. Please use pip or conda to install the needed libraries in `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```
   
2. Please install the required neural net files
   ```bash
    cd content
    gsutil -m cp -r 'gs://magentadata/models/music_transformer/checkpoints/*' .
    gsutil -q -m cp -r 'gs://magentadata/models/music_transformer/primers/*' .
    gsutil -q -m cp 'gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2' .
    ```
3. Run the `Launcher.sh` in terminal with
    ```bash
    ./Launcher.sh 
    ```
   
4. Alternatively the frontend and backend can be run on different machines with one hosting the backend server and another connecting to it with the front end code (check line 32 for correct ip address)

# Usage:
There are currently 4 available modes which can be switched between with the use of on keyboard "Dynamic Pad" aka "Drum Pad":
1. Replay: Replays what the user played
2. Quiet: Goes silent and finally gives you inner peace
3. Improv: Jams along after you play taking inspiration from what the user played
4. Accompaniment: Plays layering on top of the last thing you played



# For Docker
```bash
docker build -t musicgeneration Docker
```

```bash
sudo docker run -it --rm --runtime nvidia --network host -v "$PWD:/app" -w /app musicgeneration
```

