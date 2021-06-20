#!/usr/bin/env python3

import time
import pygame.midi
import pygame.mixer
import note_seq
from note_seq.protobuf import music_pb2
import fluidsynth
from collections import deque

import numpy as np
import tensorflow.compat.v1 as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf

from threading import Thread
import pretty_midi


tf.disable_v2_behavior()


SF2_PATH = "/Users/jjclark/Projects/music-generation/content/Yamaha-C5-Salamander-JNv5.1.sf2"
SAMPLE_RATE = 16000


event_buffer = deque()
last_event_time = 0
generated_notes = None
generated_bars = deque()

midi_in = None
midi_out = None


def main():
    start()
    while True:
        update()
        time.sleep(0.01)
    stop()


def start():
    global captured_notes
    global fs
    global midi_in
    global midi_out

    pygame.mixer.init()
    pygame.midi.init()
    midi_in_id = pygame.midi.get_default_input_id()
    if midi_in_id == -1:
        print("No MIDI input device detected")
        return -1

    midi_in = pygame.midi.Input(pygame.midi.get_default_input_id())
    midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())
    print("Connected to MIDI input", pygame.midi.get_device_info(midi_in.device_id))

    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(SF2_PATH)
    fs.program_select(0, sfid, 0, 0)

    captured_notes = music_pb2.NoteSequence()
    captured_notes.tempos.add(qpm=60)

    generation_thread = Thread(target=generate_notes_loop_unconditional, args=())
    generation_thread.start()

    playback_thread = Thread(target=play_selected_notes, args=())
    playback_thread.start()

    print("Started")


def stop():
    global fs
    pygame.midi.quit()
    fs.delete()


def update():
    global captured_notes
    global last_event_time
    global fs
    global midi_in

    new_events = midi_in.read(100)
    # if len(new_events) > 0:
    #     print(f"{len(new_events)} new events, {len(event_buffer)} buffered events")
    # event_buffer.extend(new_events)

    for i in range(len(new_events)):
        event, timestamp = new_events[i]
        last_event_time = timestamp
        event_code, pitch, velocity, _ = event

        if 144 <= event_code < 160:
            # fs.noteon(event_code - 144, pitch, velocity)
            last_event_time += 100_000
            event_buffer.append(new_events[i])

        if 128 <= event_code < 144:
            # fs.noteoff(event_code - 128, pitch)

            start_code = event_code + 16
            for j in range(len(event_buffer)):
                if event_buffer[j][0][1] == pitch and event_buffer[j][0][0] == start_code:
                    start_time = (event_buffer[j][1]) / 1000
                    end_time = (new_events[i][1]) / 1000
                    velocity = event_buffer[j][0][2]
                    note_name = pretty_midi.note_number_to_name(pitch)
                    print(f"Received input {note_name}")
                    captured_notes.notes.add(
                        pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity)
                    captured_notes.total_time = end_time
                    event_buffer.remove(event_buffer[j])
                    break

    select_notes_to_play()
    time.sleep(0.01)


class UnconditionalGenerator(object):
    def __init__(self):
        self.targets = []
        self.decode_length = 64

        model_name = 'transformer'
        hparams_set = 'transformer_tpu'
        ckpt_path = '/Users/jjclark/Projects/music-generation/content/unconditional_model_16.ckpt'

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
        print("Initialized neural net")

    def decode(self, ids, encoder):
        # Decode a list of IDs.
        ids = list(ids)
        if text_encoder.EOS_ID in ids:
            ids = ids[:ids.index(text_encoder.EOS_ID)]
        return encoder.decode(ids)

    def continuation(self, primer_ns):
        self.targets = self.unconditional_encoders['targets'].encode_note_sequence(primer_ns)

        # Remove the end token from the encoded primer.
        self.targets = self.targets[:-1]

        # if len(self.targets) >= self.decode_length / 2:
        #     self.targets = self.targets[-self.decode_length / 2:]

        # decode_length = max(0, 128 - len(targets))
        # if len(targets) >= 128:
        #     print('Primer has more events than maximum sequence length; nothing will be generated.')
        # self.decode_length = 128

        # Generate sample events.
        sample_ids = next(self.unconditional_samples)['outputs']

        # Decode to NoteSequence.
        midi_filename = self.decode(
            sample_ids,
            encoder=self.unconditional_encoders['targets'])
        ns = note_seq.midi_file_to_note_sequence(midi_filename)

        # Append continuation to primer.
        # continuation_ns = note_seq.concatenate_sequences([primer_ns, ns])

        return ns

    def generate_notes(self, captured_notes):
        print("Generating continued notes...")

        # process_captured_notes(captured_notes)
        # primer_ns = truncate_ns_right(captured_notes, 10.0)
        primer_ns = captured_notes
        generated_notes = self.continuation(primer_ns)
        return generated_notes


def generate_notes_loop_unconditional():
    global captured_notes
    global generated_bars

    generator = UnconditionalGenerator()

    while True:
        if len(generated_bars) < 3 and len(captured_notes.notes) > 5:
            gen_start = time.time()
            generated_bar = generator.generate_notes(captured_notes)
            generated_bars.append(generated_bar)
            print(f"Spent {time.time() - gen_start} sec generating {len(generated_bar.notes)} notes. {len(generated_bars)} bars in queue.")
        time.sleep(0.1)


def select_notes_to_play():
    global captured_notes
    global generated_bars
    global generated_notes
    global last_event_time

    if last_event_time == 0:
        return

    if len(generated_bars) is 0:
        return

    idle_time = (pygame.midi.time() - last_event_time) / 1000
    if idle_time < 1.0:
        interrupt_ns(generated_notes)
        generated_notes = None
        return

    if idle_time > 15.0:
        return

    if is_currently_playing(generated_notes):
        return

    # Select the next bar to be played
    interrupt_ns(generated_notes)
    generated_notes = None

    tmp_notes = generated_bars.popleft()
    if tmp_notes is None:
        return

    print(f"Playing {len(tmp_notes.notes)} generated notes. {len(generated_bars)} bars in queue")

    # if pygame.mixer.music.get_busy():
    #     pygame.mixer.music.stop()

    captured_time = max(captured_notes.total_time, pygame.midi.time() / 1000)
    generated_notes = note_seq.concatenate_sequences(
        [captured_notes, tmp_notes],
        [captured_time, tmp_notes.total_time])

    # note_seq.note_sequence_to_midi_file(generated_notes, "generated_notes.mid")
    # pygame.mixer.music.load("generated_notes.mid")
    # pygame.mixer.music.play()
    # last_event_time += pygame.midi.time() + generated_notes.total_time * 1000
    # time.sleep(0.1)
    # while pygame.mixer.music.get_busy():
    #     time.sleep(0.1)
    # generated_notes = None


def is_currently_playing(ns):
    if not ns:
        return False
    if ns.total_time < pygame.midi.time() / 1000:
        return False
    return True


def interrupt_ns(ns):
    global fs
    global midi_out
    if ns:
        for note in ns.notes:
            # print(f"stopping {pretty_midi.note_number_to_name(note.pitch)}")
            # fs.noteoff(chan=0, key=note.pitch)
            midi_out.note_off(note.pitch, velocity=note.velocity, channel=0)


def play_selected_notes():
    global fs
    global midi_out
    global generated_notes

    last_played_time = 0
    while True:
        now = pygame.midi.time() / 1000
        if generated_notes is not None:
            for note in generated_notes.notes:
                if last_played_time < note.start_time < now:
                    # print(f"playing {pretty_midi.note_number_to_name(note.pitch)}")
                    # fs.noteon(chan=0, key=note.pitch, vel=note.velocity)
                    midi_out.note_on(note.pitch, velocity=note.velocity, channel=0)
                if last_played_time < note.end_time < now:
                    # print(f"stopping {pretty_midi.note_number_to_name(note.pitch)}")
                    # fs.noteoff(chan=0, key=note.pitch)
                    midi_out.note_off(note.pitch, velocity=note.velocity, channel=0)
        last_played_time = now
        time.sleep(0.01)


def process_captured_notes(captured_notes: music_pb2.NoteSequence):
    if len(captured_notes.notes) == 0:
        return

    t0 = min(note.start_time for note in captured_notes.notes)
    for note in captured_notes.notes:
        note.start_time -= t0
        note.end_time -= t0

    captured_notes.total_time = max(note.end_time for note in captured_notes.notes)

    # captured_notes.tempos.add(qpm=60)

    # note_seq.fluidsynth(captured_notes, 44000, sf2_path="./content/Yamaha-C5-Salamander-JNv5.1.sf2")


def truncate_ns_left(ns: music_pb2.NoteSequence, end_time: float):
    new_ns = music_pb2.NoteSequence()
    for note in reversed(ns.notes):
        if note.start_time > end_time:
            ns.notes.remove(note)
    return new_ns


def truncate_ns_right(ns: music_pb2.NoteSequence, time_from_end: float):
    # if len(ns.notes) <= time_from_end:
    #     return ns
    # new_ns = music_pb2.NoteSequence()
    # for note in ns.notes[-int(time_from_end):]:
    #     new_ns.notes.append(note)
    # process_captured_notes(new_ns)
    # return new_ns

    end_time = max(note.end_time for note in ns.notes)
    cutoff = end_time - time_from_end
    new_ns = music_pb2.NoteSequence()
    for note in ns.notes:
        if note.end_time > cutoff:
            new_ns.notes.append(note)

    process_captured_notes(new_ns)
    return new_ns


if __name__ == "__main__":
    retcode = main()
    exit(retcode)

# agenda for next time:
# - load this module in notebook
# - make the main loop faster
