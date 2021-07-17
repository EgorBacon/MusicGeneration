#!/usr/bin/env python3

import os
from flask import Flask, request, redirect, send_from_directory, send_file
from werkzeug.utils import secure_filename
from generators import UnconditionalGenerator, MelodyConditionedGenerator
from note_seq.midi_io import note_sequence_to_midi_file, midi_file_to_note_sequence
from datetime import datetime

def stamped_filename(filename):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%H%M%S-")
    return secure_filename(timestampStr + filename)

def create_app(test_config=None):
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = './uploads'
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'])
    except FileExistsError:
        pass

    @app.route('/', methods=['GET'])
    def manual_upload():
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form action="/generate_unchanged" method="post" enctype="multipart/form-data">
          <fieldset>
            <legend>Generate Unchanged</legend>
            <input type="file" name="file">
            <input type="submit" value="Upload">
          </fieldset>
        </form>
        <form action="/generate_unconditional" method="post" enctype="multipart/form-data">
          <fieldset>
            <legend>Generate Unconditional</legend>
            <input type="file" name="file">
            <input type="submit" value="Upload">
          </fieldset>
        </form>
        <form action="/generate_melody_conditioned" method="post" enctype="multipart/form-data">
          <fieldset>
            <legend>Generate Accompaniment</legend>
            <input type="file" name="file">
            <input type="submit" value="Upload">
          </fieldset>
        </form>
        '''

    @app.route('/generate_unchanged', methods=['POST'])
    def generate_unchanged():
        file = request.files['file']
        if file.filename == '':
            # empty file
            return redirect("/")
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    uc_generator = UnconditionalGenerator()

    @app.route('/generate_unconditional', methods=['POST'])
    def generate_unconditional():
        file = request.files['file']
        if file.filename == '':
            # empty file
            return redirect("/")
        filename = stamped_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        generated_path = os.path.join(app.config['UPLOAD_FOLDER'], stamped_filename("generated_notes.mid"))
        captured_notes = midi_file_to_note_sequence(file_path)
        generated_notes = uc_generator.generate_notes(captured_notes)
        note_sequence_to_midi_file(generated_notes, generated_path)

        return send_file(generated_path)

    mc_generator = MelodyConditionedGenerator()

    @app.route('/generate_melody_conditioned', methods=['POST'])
    def generate_melody_conditioned():
        file = request.files['file']
        if file.filename == '':
            # empty file
            return redirect("/")
        filename = stamped_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        generated_path = os.path.join(app.config['UPLOAD_FOLDER'], stamped_filename("generated_notes.mid"))
        captured_notes = midi_file_to_note_sequence(file_path)
        generated_notes = mc_generator.generate_notes(captured_notes)
        note_sequence_to_midi_file(generated_notes, generated_path)

        return send_file(generated_path)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
