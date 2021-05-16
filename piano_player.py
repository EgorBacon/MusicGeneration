import time
import fluidsynth
import pygame.midi
from note_seq.protobuf import music_pb2
import threading
from collections import deque

import numpy as np
import os
import tensorflow.compat.v1 as tf

#from google.colab import files

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf
import note_seq

tf.disable_v2_behavior()



event_buffer = deque()

SF2_PATH = './content/Yamaha-C5-Salamander-JNv5.1.sf2'
fs = fluidsynth.Synth()

last_event_time = 0

def main():
    pygame.midi.init()
    midi_in = pygame.midi.Input(pygame.midi.get_default_input_id())

    midi_in = pygame.midi.Input(1)

    fs.start()

    sfid = fs.sfload(SF2_PATH)
    fs.program_select(0, sfid, 0, 0)

    captured_notes = music_pb2.NoteSequence()

    print("Now Playing...")

    try:
        while True:
            time.sleep(0.01)
            interaction_loop(midi_in, fs, captured_notes)
    except KeyboardInterrupt:
        print("Exiting")
        
    pygame.midi.quit()
    fs.delete()




def interaction_loop(midi_in, fs, captured_notes):
    global last_event_time

    new_events = midi_in.read(100)


    if len(new_events) > 0:
    	print(len(new_events), " new events", len(event_buffer))   

    for i in range(len(new_events)):
        event,timestamp = new_events[i]
        last_event_time = timestamp
        event_code, pitch, velocity, _ = event
        if 144 <= event_code < 160:
            fs.noteon(event_code - 144, pitch, velocity)
            last_event_time += 10000
            event_buffer.append(new_events[i])
        if 128 <= event_code < 144:
            fs.noteoff(event_code - 128, pitch)
            start_code = event_code + 16
            for j in range (len(event_buffer)):
                if event_buffer[j][0][1] == pitch and event_buffer[j][0][0] == start_code:
                    captured_notes.notes.add(
                        pitch=pitch,
                        start_time=(event_buffer[j][1]) / 1000,
                        end_time=(new_events[i][1]) / 1000,
                        velocity=event_buffer[j][0][2])

                    event_buffer.remove(event_buffer[j])
                    break

    if pygame.midi.time() - last_event_time > 500:
    	generate_notes(fs, captured_notes)

def generate_notes(fs, captured_notes):
	print("replaying what you played...")
	
	if len(captured_notes) == 0:
		return

	process_captured_notes(captured_notes)

	note_seq.note_sequence_to_midi_file(captured_notes, "captured_notes.mid")



def process_captured_notes(captured_notes):
    t0 = captured_notes.notes[0].start_time
    for note in captured_notes.notes:
        note.start_time -= t0
        note.end_time -= t0
    
    captured_notes.total_time = max(note.end_time for note in captured_notes.notes)

    captured_notes.tempos.add(qpm = 60)

main()

