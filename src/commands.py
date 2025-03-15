from typing import Any
from .audio_processor import AudioProcessor

class Command:
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        raise NotImplementedError

class LoadCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.load_wav(args.input_file)

class SaveCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.save_wav(args.output_file)

class ConvertToMonoCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.convert_stereo_to_mono()

class TrimSilenceCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.trim_silence(args.silence_thresh, args.keep_silence)

class SplitOnSilenceCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.split_on_silence(args.output_dir, args.min_silence_len, args.silence_thresh, args.keep_silence)

class LoopAudioCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.loop_audio(args.num_loops)

class CutSelectionCommand(Command):
    def execute(self, processor: AudioProcessor, args: Any) -> None:
        processor.cut_selection(args.start_time, args.end_time)