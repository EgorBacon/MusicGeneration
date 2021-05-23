#!/usr/bin/env python3

import time
import pygame.midi
import pygame.mixer
import note_seq
from note_seq.protobuf import music_pb2
import fluidsynth
from collections import deque

import numpy as np
import os
import tensorflow.compat.v1 as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf


tf.disable_v2_behavior()
print('Done loading libraries')


SF2_PATH = "/Users/jjclark/Projects/music-generation/content/Yamaha-C5-Salamander-JNv5.1.sf2"
SAMPLE_RATE = 16000


event_buffer = deque()
last_event_time = 0

targets = []
decode_length = 0


def load_unconditional_model():
    global unconditional_encoders
    global unconditional_samples

    model_name = 'transformer'
    hparams_set = 'transformer_tpu'
    ckpt_path = '/Users/jjclark/Projects/music-generation/content/unconditional_model_16.ckpt'

    class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
        @property
        def add_eos_symbol(self):
            return True

    problem = PianoPerformanceLanguageModelProblem()
    unconditional_encoders = problem.get_feature_encoders()

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
        global targets
        global decode_length
        while True:
            yield {
                'targets': np.array([targets], dtype=np.int32),
                'decode_length': np.array(decode_length, dtype=np.int32)
            }

    # Start the Estimator, loading from the specified checkpoint.
    input_fn = decoding.make_input_fn_from_generator(input_generator())
    unconditional_samples = estimator.predict(
        input_fn, checkpoint_path=ckpt_path)

    # "Burn" one.
    _ = next(unconditional_samples)


# Decode a list of IDs.
def decode(ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
        ids = ids[:ids.index(text_encoder.EOS_ID)]
    return encoder.decode(ids)


def continuation(primer_ns):
    global targets
    global decode_length
    global unconditional_encoders
    global unconditional_samples

    targets = unconditional_encoders['targets'].encode_note_sequence(primer_ns)

    # Remove the end token from the encoded primer.
    targets = targets[:-1]

    # decode_length = max(0, 128 - len(targets))
    # if len(targets) >= 128:
    #     print('Primer has more events than maximum sequence length; nothing will be generated.')
    decode_length = 128

    # Generate sample events.
    sample_ids = next(unconditional_samples)['outputs']

    # Decode to NoteSequence.
    midi_filename = decode(
        sample_ids,
        encoder=unconditional_encoders['targets'])
    ns = note_seq.midi_file_to_note_sequence(midi_filename)

    # Append continuation to primer.
    # continuation_ns = note_seq.concatenate_sequences([primer_ns, ns])

    return ns


def main():
    global captured_notes
    load_unconditional_model()

    pygame.mixer.init()
    pygame.midi.init()
    midi_in_id = pygame.midi.get_default_input_id()
    if midi_in_id == -1:
        print("No MIDI input device detected")
        return -1

    midi_in = pygame.midi.Input(pygame.midi.get_default_input_id())
    print("Connected to MIDI input", pygame.midi.get_device_info(midi_in.device_id))

    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(SF2_PATH)
    fs.program_select(0, sfid, 0, 0)

    captured_notes = music_pb2.NoteSequence()

    print("Starting")
    try:
        while True:
            interaction_loop(midi_in, fs)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    print("Exiting")
    pygame.midi.quit()
    fs.delete()


def interaction_loop(midi_in, fs):
    global captured_notes
    global last_event_time

    new_events = midi_in.read(100)
    if len(new_events) > 0:
        print(f"{len(new_events)} new events, {len(event_buffer)} buffered events")
    # event_buffer.extend(new_events)

    for i in range(len(new_events)):
        event, timestamp = new_events[i]
        last_event_time = timestamp
        event_code, pitch, velocity, _ = event

        if 144 <= event_code < 160:
            fs.noteon(event_code - 144, pitch, velocity)
            last_event_time += 100000
            event_buffer.append(new_events[i])

        if 128 <= event_code < 144:
            fs.noteoff(event_code - 128, pitch)

            start_code = event_code + 16
            for j in range(len(event_buffer)):
                if event_buffer[j][0][1] == pitch and event_buffer[j][0][0] == start_code:
                    captured_notes.notes.add(
                        pitch=pitch,
                        start_time=(event_buffer[j][1]) / 1000,
                        end_time=(new_events[i][1]) / 1000,
                        velocity=event_buffer[j][0][2])
                    event_buffer.remove(event_buffer[j])
                    break

    idle_time = pygame.midi.time() - last_event_time
    if 2000 < idle_time < 3000:
        generate_notes(fs)
        captured_notes = music_pb2.NoteSequence()


def generate_notes(fs):
    if len(captured_notes.notes) == 0:
        return

    print("Continuing...")

    process_captured_notes(captured_notes)
    primer_ns = truncate_ns_right(captured_notes, 10.0)
    continued_notes = continuation(primer_ns)

    note_seq.note_sequence_to_midi_file(continued_notes, "generated_notes.mid")
    pygame.mixer.music.load("generated_notes.mid")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def process_captured_notes(captured_notes: music_pb2.NoteSequence):
    if len(captured_notes.notes) == 0:
        return

    t0 = min(note.start_time for note in captured_notes.notes)
    for note in captured_notes.notes:
        note.start_time -= t0
        note.end_time -= t0

    captured_notes.total_time = max(note.end_time for note in captured_notes.notes)

    captured_notes.tempos.add(qpm=60)

    # note_seq.fluidsynth(captured_notes, 44000, sf2_path="./content/Yamaha-C5-Salamander-JNv5.1.sf2")


def truncate_ns_left(ns: music_pb2.NoteSequence, end_time: float):
    new_ns = music_pb2.NoteSequence()
    for note in reversed(ns.notes):
        if note.start_time > end_time:
            ns.notes.remove(note)
    return new_ns


def truncate_ns_right(ns: music_pb2.NoteSequence, time_from_end: float):
    new_ns = music_pb2.NoteSequence()
    cutoff = ns.total_time - time_from_end
    for note in ns.notes:
        if note.end_time > cutoff:
            new_ns.notes.append(note)

    process_captured_notes(new_ns)
    return new_ns



def main2():
    # I don't understand why I can hear this but not the version in interaction_loop
    import time
    import fluidsynth
    fs = fluidsynth.Synth()
    fs.start()

    sfid = fs.sfload(SF2_PATH)
    fs.program_select(0, sfid, 0, 0)

    fs.noteon(0, 60, 30)
    fs.noteon(0, 67, 30)
    fs.noteon(0, 76, 30)

    time.sleep(1.0)

    fs.noteoff(0, 60)
    fs.noteoff(0, 67)
    fs.noteoff(0, 76)

    time.sleep(1.0)


if __name__ == "__main__":
    retcode = main()
    exit(retcode)

# agenda for next time:
# - load this module in notebook
# - make the main loop faster
