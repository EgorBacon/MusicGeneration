import time
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import sys
sys.modules["note_seq.audio_io"] = {}
import note_seq
import numpy as np

def import_submodule(module_name):
    import sys
    from pathlib import Path
    import importlib
    parts = module_name.split(".")
    package_name = parts[0]
    package_spec = importlib.util.find_spec(package_name)
    sys.modules[package_name] = importlib.util.module_from_spec(package_spec)
    submodule_path = Path(package_spec.submodule_search_locations[0])/Path(*parts[1:]) #/"__init__.py"
    if submodule_path.is_dir():
        submodule_path = submodule_path/"__init__.py"
    else :
        submodule_path = submodule_path.with_suffix(".py")
    submodule_spec = importlib.util.spec_from_file_location(module_name, submodule_path)
    submodule_module = importlib.util.module_from_spec(submodule_spec)
    sys.modules[module_name] = submodule_module
    submodule_spec.loader.exec_module(submodule_module)

    return submodule_module

import_submodule("tensor2tensor.data_generators")
import_submodule("tensor2tensor.models.transformer")
import_submodule("magenta.models.score2perf")

from magenta.models.score2perf import score2perf
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import trainer_lib, decoding




import os
#from magenta.models.performance_rnn import performance_sequence_generator
#from magenta.models.shared import sequence_generator_bundle
#from note_seq.protobuf import generator_pb2
#from note_seq.protobuf import music_pb2


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

        return self.continuation(primer_ns)


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

class RandomGenerator(object):
    def __init__(self):
        pass

    def generate_notes(self, captured_notes):
        print("Generating random notes...")
        new_notes = note_seq.NoteSequence()
        # - identify events in captured_notes
        # - for each type of event, make a list of all the type of events that come after it
        # - several times:
            # - look at last note played
            # - select a likely event to follow
        return new_notes


"""class PerfomanceWithDynamicsGenerator:

    def __init__(self):
        MODEL_NAME = 'performance_with_dynamics'
        BUNDLE_NAME = MODEL_NAME + '.mag'

        bundle = sequence_generator_bundle.read_bundle_file(os.path.join('./content', BUNDLE_NAME))
        generator_map = performance_sequence_generator.get_generator_map()
        self.generator = generator_map[MODEL_NAME](checkpoint=None, bundle=bundle)
        self.generator.initialize()


    def generate_notes(self, melody_ns):
        generator_options = generator_pb2.GeneratorOptions()
        generator_options.args['temperature'].float_value = 1.0  # Higher is more random; 1.0 is default.
        generate_section = generator_options.generate_sections.add(start_time=0, end_time=30)
        sequence = self.generator.generate(music_pb2.NoteSequence(), generator_options)

        return sequence
"""
