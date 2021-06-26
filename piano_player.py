#!/usr/bin/env python3.7
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
SAMPLE_RATE = 16000
fs = fluidsynth.Synth()

last_event_time = 0


generated_notes = None
generated_bars = deque()
selected_generator = "unconditional"


class UnconditionalGenerator:
    targets = []
    decode_length = 0

    def decode(self, ids, encoder):
        ids = list(ids)
        if text_encoder.EOS_ID in ids:
            ids = ids[:ids.index(text_encoder.EOS_ID)]
        return encoder.decode(ids)

    def __init__(self):
        #@title Setup and Load Checkpoint
        #@markdown Set up generation from an unconditional Transformer
        #@markdown model.

        model_name = 'transformer'
        hparams_set = 'transformer_tpu'
        ckpt_path = './content/unconditional_model_16.ckpt'
        #ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/unconditional_model_16.ckpt'

        class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
          @property
          def add_eos_symbol(self):
            return True

        problem = PianoPerformanceLanguageModelProblem()
        self.unconditional_encoders = problem.get_feature_encoders()

        # Set up HParams.
        hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
        trainer_lib.add_problem_hparams(hparams, problem)
        hparams.num_hidden_layers = 16
        hparams.sampling_method = 'random'

        # Set up decoding HParams.
        decode_hparams = decoding.decode_hparams()
        decode_hparams.alpha = 0.0
        decode_hparams.beam_size = 1

        # Create Estimator.
        run_config = trainer_lib.create_run_config(hparams)
        estimator = trainer_lib.create_estimator(
            model_name, hparams, run_config,
            decode_hparams=decode_hparams)

        # Create input generator (so we can adjust priming and
        # decode length on the fly).
        def input_generator():
            while True:
                yield {
                    'targets': np.array([self.targets], dtype=np.int32),
                    'decode_length': np.array(self.decode_length, dtype=np.int32)
                }


        # Start the Estimator, loading from the specified checkpoint.
        input_fn = decoding.make_input_fn_from_generator(input_generator())
        self.unconditional_samples = estimator.predict(
            input_fn, checkpoint_path=ckpt_path)

        # "Burn" one.
        _ = next(self.unconditional_samples)

        print("UnconditionalGenerator init finished")

    def continuation(self, primer_ns):
        #@title Generate Continuation
        #@markdown Continue a piano performance, starting with the
        #@markdown chosen priming sequence.

        self.targets = self.unconditional_encoders['targets'].encode_note_sequence(primer_ns)

        # Remove the end token from the encoded primer.
        self.targets = self.targets[:-1]

        self.decode_length = 64

        # Generate sample events.
        sample_ids = next(self.unconditional_samples)['outputs']

        # Decode to NoteSequence.
        midi_filename = self.decode(
            sample_ids,
            encoder=self.unconditional_encoders['targets'])
        ns = note_seq.midi_file_to_note_sequence(midi_filename)

        # return continuation ns
        return ns

    def generate_notes(self, primer_ns):
        global captured_notes
        global last_event_time
        global generated_bars

        print("generating notes")

        #primer_ns = captured_notes

        gen_start = time.time()
        generated_bars.append(self.continuation(primer_ns))
        total_time = time.time() - gen_start

        print(f"Generated {len(generated_bars)} bars which took {total_time} sec")    

class MelodyConditionedGenerator:

    def decode(self, ids, encoder):
        ids = list(ids)
        if text_encoder.EOS_ID in ids:
            ids = ids[:ids.index(text_encoder.EOS_ID)]
        return encoder.decode(ids)


    def __init__(self):
        model_name = 'transformer'
        hparams_set = 'transformer_tpu'
        ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/melody_conditioned_model_16.ckpt'

        class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
          @property
          def add_eos_symbol(self):
            return True

        problem = MelodyToPianoPerformanceProblem()
        self.melody_conditioned_encoders = problem.get_feature_encoders()

        # Set up HParams.
        hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
        trainer_lib.add_problem_hparams(hparams, problem)
        hparams.num_hidden_layers = 16
        hparams.sampling_method = 'random'

        # Set up decoding HParams.
        decode_hparams = decoding.decode_hparams()
        decode_hparams.alpha = 0.0
        decode_hparams.beam_size = 1

        # Create Estimator.
        run_config = trainer_lib.create_run_config(hparams)
        estimator = trainer_lib.create_estimator(
            model_name, hparams, run_config,
            decode_hparams=decode_hparams)

        # These values will be changed by the following cell.
        self.inputs = []
        self.decode_length = 0

        # Create input generator.
        def input_generator():
          global inputs
          while True:
            yield {
                'inputs': np.array([[self.inputs]], dtype=np.int32),
                'targets': np.zeros([1, 0], dtype=np.int32),
                'decode_length': np.array(self.decode_length, dtype=np.int32)
            }

        # Start the Estimator, loading from the specified checkpoint.
        input_fn = decoding.make_input_fn_from_generator(input_generator())
        self.melody_conditioned_samples = estimator.predict(
            input_fn, checkpoint_path=ckpt_path)

        # "Burn" one.
        _ = next(self.melody_conditioned_samples)

        print("melody conditioned generator initialised")

    def generate_notes(self, melody_ns):
        self.inputs = self.melody_conditioned_encoders['inputs'].encode_note_sequence(
      melody_ns)


        self.decode_length = 256
        sample_ids = next(self.melody_conditioned_samples)['outputs']

        # Decode to NoteSequence.
        midi_filename = self.decode(
            sample_ids,
            encoder=self.melody_conditioned_encoders['targets'])
        accompaniment_ns = note_seq.midi_file_to_note_sequence(midi_filename)

        # Play and plot.
        return accompaniment_ns






def main():
    start()
    while True:
        update()
        time.sleep(0.01)
    stop()

def start():
    global fs
    global midi_in
    global captured_notes
    pygame.midi.init()
    midi_in = pygame.midi.Input(pygame.midi.get_default_input_id())

    midi_in = pygame.midi.Input(1)

    pygame.mixer.init()

    fs.start()

    sfid = fs.sfload(SF2_PATH)
    fs.program_select(0, sfid, 0, 0)

    captured_notes = music_pb2.NoteSequence()


    generation_thread = threading.Thread(target = generate_notes_loop_unconditional, args = (fs,))
    generation_thread.start()

    play_selective_notes_thread = threading.Thread(target = play_selective_notes)
    play_selective_notes_thread.start()

    print("Now Playing...")


def stop():
    global fs
    pygame.midi.quit()
    fs.delete()



def update():
    global last_event_time
    global captured_notes
    global fs
    global midi_in
    global selected_generator

    new_events = midi_in.read(100)


    if len(new_events) > 0:
        print(len(new_events), " new events", len(event_buffer))   


    for i in range(len(new_events)):
        event,timestamp = new_events[i]
        last_event_time = timestamp
        event_code, pitch, velocity, _ = event
        #print(event)
        if 144 <= event_code < 160:
            fs.noteon(event_code - 144, pitch, velocity)

            last_event_time += 10000
            event_buffer.append(new_events[i])
        elif 128 <= event_code < 144:
            fs.noteoff(event_code - 128, pitch)
            start_code = event_code + 16
            for j in range (len(event_buffer)):
                if event_buffer[j][0][1] == pitch and event_buffer[j][0][0] == start_code:
                    start_time = (event_buffer[j][1]) / 1000
                    end_time = (new_events[i][1]) / 1000
                    velocity = event_buffer[j][0][2]

                    captured_notes.notes.add(
                        pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity)
                    captured_notes.total_time = end_time

                    event_buffer.remove(event_buffer[j])
                    break

        elif 160 <= event_code < 176:
            if pitch == 36:
                selected_generator = "unconditional"
            elif pitch == 38:
                selected_generator = "melody_conditioned"


    select_notes_to_play()
    time.sleep(0.01)

# Decode a list of IDs.

  



def generate_notes_loop_unconditional(fs):
    generators = {
        "unconditional" : UnconditionalGenerator(),
        "melody_conditioned" : MelodyConditionedGenerator()
        }


    while True:
        if len(captured_notes.notes) < 5:
            continue

        if len(generated_bars) > 3:
            continue

        print("using generator " + selected_generator)
        generator = generators[selected_generator]
        input_ns = truncate_right_ns(captured_notes, 30.0)
        generated_bar = generator.generate_notes(input_ns)
        generated_bars.append(generated_bar)
        time.sleep(0.1)



def select_notes_to_play():
    global generated_bars
    global captured_notes
    global generated_notes

    if len(generated_bars) == 0:
        return


    idle_time = pygame.midi.time() - last_event_time

    if idle_time < 1000:
        interrupt_playing(generated_notes)
        generated_notes = None
        return
    if idle_time > 15000:
        return
    if is_currently_playing(generated_notes):
        return
    if generated_notes != None:
        interrupt_playing(generated_notes)

    generated_notes = generated_bars.popleft()

    captured_time = max(pygame.midi.time() / 1000, captured_notes.total_time)

    generated_notes = note_seq.concatenate_sequences([captured_notes, generated_notes],[captured_time, generated_notes.total_time])

    print(f"Now Playing...{len(generated_notes.notes)}")

def interrupt_playing(ns):
    global fs
    if ns is not None:
        for note in ns.notes:
            fs.noteoff(0, note.pitch)

def is_currently_playing(ns):
    if not ns:
        return False
    if ns.total_time< pygame.midi.time() / 1000:
        return False
    return True

def play_selective_notes():
    global generated_notes
    global fs

    last_played_time = pygame.midi.time()
    while True:
        now = pygame.midi.time() / 1000
        if generated_notes != None:
            for note in generated_notes.notes:
                if last_played_time < note.start_time < now:
                    fs.noteon(0, note.pitch, note.velocity)

                if last_played_time < note.end_time < now:
                    fs.noteoff(0, note.pitch)

        last_played_time = now
        time.sleep(0.05)  


def truncate_left_ns(ns, end_time):
    for note in reversed(ns.notes):
        if note.start_time > end_time:
            ns.notes.remove(note)

def truncate_right_ns(ns, time_from_end):
    if len(ns.notes) == 0:
        return ns

    end_time = max(note.end_time for note in ns.notes)
    cutoff = end_time - time_from_end
    new_ns = music_pb2.NoteSequence()
    for note in ns.notes:
        if note.end_time > cutoff:
            new_ns.notes.append(note)
    return new_ns

def process_captured_notes(captured_notes):
    t0 = captured_notes.notes[0].start_time
    for note in captured_notes.notes:
        note.start_time -= t0
        note.end_time -= t0
    
    captured_notes.total_time = max(note.end_time for note in captured_notes.notes)

    captured_notes.tempos.add(qpm = 60)

if __name__ == "__main__":
    main()

