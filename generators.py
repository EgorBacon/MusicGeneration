import numpy as np
from note_seq.midi_io import midi_file_to_note_sequence
from magenta.models.score2perf import score2perf
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import trainer_lib, decoding

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class UnconditionalGenerator(object):
    def __init__(self):
        self.targets = []
        self.decode_length = 64

        model_name = 'transformer'
        hparams_set = 'transformer_tpu'
        ckpt_path = './content/unconditional_model_16.ckpt'

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
        print("Initialized unconditional neural net")

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
        ns = midi_file_to_note_sequence(midi_filename)

        # Append continuation to primer.
        # continuation_ns = concatenate_sequences([primer_ns, ns])

        return ns

    def generate_notes(self, captured_notes):
        print("Generating continued notes...")

        # process_captured_notes(captured_notes)
        # primer_ns = truncate_ns_right(captured_notes, 10.0)
        primer_ns = captured_notes
        generated_notes = self.continuation(primer_ns)
        return generated_notes


class MelodyConditionedGenerator(object):
    def __init__(self):
        model_name = 'transformer'
        hparams_set = 'transformer_tpu'
        ckpt_path = './content/melody_conditioned_model_16.ckpt'

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
        self.decode_length = 256

        # Create input generator.
        def input_generator():
            while True:
                yield {
                    'inputs': np.array([[self.inputs]], dtype=np.int32),
                    'targets': np.zeros([1, 0], dtype=np.int32),
                    'decode_length': np.array(self.decode_length, dtype=np.int32)
                }

        # Start the Estimator, loading from the specified checkpoint.
        input_fn = decoding.make_input_fn_from_generator(input_generator())
        self.melody_conditioned_samples = estimator.predict(input_fn, checkpoint_path=ckpt_path)

        # "Burn" one.
        _ = next(self.melody_conditioned_samples)
        print("Initialized melody-conditioned neural net")

    def decode(self, ids, encoder):
        # Decode a list of IDs.
        ids = list(ids)
        if text_encoder.EOS_ID in ids:
            ids = ids[:ids.index(text_encoder.EOS_ID)]
        return encoder.decode(ids)

    def generate_notes(self, melody_ns):
        self.inputs = self.melody_conditioned_encoders['inputs'].encode_note_sequence(melody_ns)

        # Generate sample events.
        self.decode_length = 4096
        sample_ids = next(self.melody_conditioned_samples)['outputs']

        # Decode to NoteSequence.
        midi_filename = self.decode(
            sample_ids,
            encoder=self.melody_conditioned_encoders['targets'])
        accompaniment_ns = midi_file_to_note_sequence(midi_filename)

        return accompaniment_ns