#!/usr/bin/env python3

import os
from flask import Flask, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from Generators import UnconditionalGenerator, MelodyConditionedGenerator
import note_seq

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
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        generated_path = os.path.join(app.config['UPLOAD_FOLDER'], "generated_notes.mid")

        captured_notes = note_seq.midi_file_to_note_sequence(file_path)
        generated_notes = uc_generator.generate_notes(captured_notes)
        note_seq.note_sequence_to_midi_file(generated_notes, generated_path)

        return send_from_directory(app.config['UPLOAD_FOLDER'], "generated_notes.mid")

    mc_generator = MelodyConditionedGenerator()

    @app.route('/generate_melody_conditioned', methods=['POST'])
    def generate_melody_conditioned():
        file = request.files['file']
        if file.filename == '':
            # empty file
            return redirect("/")
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        generated_path = os.path.join(app.config['UPLOAD_FOLDER'], "generated_notes.mid")

        captured_notes = note_seq.midi_file_to_note_sequence(file_path)
        generated_notes = mc_generator.generate_notes(captured_notes)
        note_seq.note_sequence_to_midi_file(generated_notes, generated_path)

        return send_from_directory(app.config['UPLOAD_FOLDER'], "generated_notes.mid")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
