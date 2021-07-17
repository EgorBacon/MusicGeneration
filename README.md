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
2. Run the `Launcher.sh` in terminal with
    ```bash
    ./Launcher.sh 
    ```

# Usage:
There are currently 4 available modes which can be switched between with the use of on keyboard "Dynamic Pad" aka "Drum Pad":
1. Replay: Replays what the user played
2. Quiet: Goes silent and finally gives you inner peace
3. Improv: Jams along after you play taking inspiration from what the user played
4. Accompaniment: Plays layering on top of the last thing you played

