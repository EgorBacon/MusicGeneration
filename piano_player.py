#!/usr/bin/env python3

import time
import pygame.midi
import pygame.mixer
import note_seq
from note_seq.protobuf import music_pb2
import fluidsynth
from collections import deque

from threading import Thread
import pretty_midi

import requests


SF2_PATH = "/Users/jjclark/Projects/music-generation/content/Yamaha-C5-Salamander-JNv5.1.sf2"
SAMPLE_RATE = 16000


event_buffer = deque()
last_event_time = 0
generated_notes = None
generated_bars = deque()
selected_generator = "unconditional"

midi_in = None
midi_out = None

STOPPING = False

def main():
    global STOPPING
    start()
    try:
        while True:
            update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        STOPPING = True
        time.sleep(0.1)
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

    generation_thread = Thread(target=generate_notes_loop, args=())
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
            note_name = pretty_midi.note_number_to_name(pitch)
            print(f"Received input {note_name}")
            # fs.noteon(event_code - 144, pitch, velocity)
            last_event_time += 100_000
            event_buffer.append(new_events[i])

        elif 128 <= event_code < 144:
            # fs.noteoff(event_code - 128, pitch)

            start_code = event_code + 16
            for j in range(len(event_buffer)):
                if event_buffer[j][0][1] == pitch and event_buffer[j][0][0] == start_code:
                    start_time = (event_buffer[j][1]) / 1000
                    end_time = (new_events[i][1]) / 1000
                    velocity = event_buffer[j][0][2]
                    captured_notes.notes.add(
                        pitch=pitch, start_time=start_time, end_time=end_time, velocity=velocity)
                    captured_notes.total_time = end_time
                    event_buffer.remove(event_buffer[j])
                    break

        elif 192 <= event_code < 208:
            _, program_no, _, _ = event
            if program_no == 1:
                change_selected_generator("quiet")
            if program_no == 2:
                change_selected_generator("identical")
            if program_no == 3:
                change_selected_generator("unconditional")
            elif program_no == 4:
                change_selected_generator("melody_conditioned")
            elif program_no == 5:
                change_selected_generator("performance")

    select_notes_to_play()
    time.sleep(0.01)


def change_selected_generator(new_selection):
    global selected_generator
    if new_selection != selected_generator:
        selected_generator = new_selection
    generated_bars.clear()
    print(f"Selected {new_selection} generator")

def generate_notes_loop():
    global captured_notes
    global generated_bars

    generators = {
        "quiet": None,
        "identical": "/generate_unchanged",
        "unconditional": "/generate_unconditional",
        "melody_conditioned": "/generate_melody_conditioned",
        "performance": "/generate_performance"
    }

    while True:
        if STOPPING:
            return
        time.sleep(0.1)
        if len(captured_notes.notes) < 6:
            continue
        if len(generated_bars) >= 3:
            continue
        generator = generators[selected_generator]
        if generator is None:
            continue
        gen_start = time.time()
        input_ns = truncate_ns_right(captured_notes, 30.0)
        generated_bar = generate_from_server(generator, input_ns)
        generated_bars.append(generated_bar)
        print(f"Spent {time.time() - gen_start:.2f} sec generating {len(generated_bar.notes)} notes. {len(generated_bars)} bars in queue.")


def generate_from_server(api, input_ns):
    captured_notes_path = "captured_notes.mid"
    note_seq.note_sequence_to_midi_file(input_ns, captured_notes_path)

    url = 'http://localhost:5000'+api
    files = {'file': open(captured_notes_path, 'rb')}
    r = requests.post(url, files=files)

    response_notes_path = "generated_notes.mid"
    with open(response_notes_path, 'wb') as f:
        f.write(r.content)
    response_notes = note_seq.midi_file_to_note_sequence(response_notes_path)
    return response_notes


auto_played_bars = 0


def select_notes_to_play():
    global captured_notes
    global generated_bars
    global generated_notes
    global auto_played_bars

    if last_event_time == 0:
        return

    if len(generated_bars) is 0:
        return

    idle_time = (pygame.midi.time() - last_event_time) / 1000
    if idle_time < 1.0:
        interrupt_ns(generated_notes)
        generated_notes = None
        return

    if idle_time < 3.50 and selected_generator == 'melody_conditioned':
        return

    if idle_time > 15.0:
        auto_played_bars = 0
        return

    # don't interrupt a fresh bar
    if is_currently_playing(generated_notes) and auto_played_bars > 0:
        return

    # Select the next bar to be played
    interrupt_ns(generated_notes)
    generated_notes = None

    tmp_notes = generated_bars.pop()
    if tmp_notes is None:
        return
    if auto_played_bars == 0:
        generated_bars.clear()

    auto_played_bars += 1

    print(f"Playing {len(tmp_notes.notes)} generated notes. {len(generated_bars)} bars in queue")
    process_captured_notes(tmp_notes)

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
        if STOPPING:
            return
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
