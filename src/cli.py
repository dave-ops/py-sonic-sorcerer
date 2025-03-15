import argparse
from typing import Dict
from .audio_processor import AudioProcessor
from .commands import (
    LoadCommand, SaveCommand, ConvertToMonoCommand,
    TrimSilenceCommand, SplitOnSilenceCommand,
    LoopAudioCommand, CutSelectionCommand
)

class SoundEditorCLI:
    def __init__(self):
        self.processor = AudioProcessor()
        self.commands: Dict[str, Command] = {
            'load': LoadCommand(),
            'save': SaveCommand(),
            'convert_to_mono': ConvertToMonoCommand(),
            'trim_silence': TrimSilenceCommand(),
            'split_on_silence': SplitOnSilenceCommand(),
            'loop_audio': LoopAudioCommand(),
            'cut_selection': CutSelectionCommand(),
        }

    def run(self):
        parser = argparse.ArgumentParser(description="CLI-based Sound Editor")
        subparsers = parser.add_subparsers(dest='command', required=True)

        # Load command
        load_parser = subparsers.add_parser('load', help='Load a WAV file')
        load_parser.add_argument('input_file', help='Path to the input WAV file')

        # Save command
        save_parser = subparsers.add_parser('save', help='Save the current audio data')
        save_parser.add_argument('output_file', help='Path to save the output WAV file')

        # Convert to mono command
        subparsers.add_parser('convert_to_mono', help='Convert stereo audio to mono')

        # Trim silence command
        trim_parser = subparsers.add_parser('trim_silence', help='Trim leading and trailing silence')
        trim_parser.add_argument('--silence_thresh', type=int, default=1000, help='Silence threshold')
        trim_parser.add_argument('--keep_silence', type=int, default=100, help='Amount of silence to keep')

        # Split on silence command
        split_parser = subparsers.add_parser('split_on_silence', help='Split audio into chunks based on silence')
        split_parser.add_argument('output_dir', help='Directory to save the output chunks')
        split_parser.add_argument('--min_silence_len', type=int, default=1000, help='Minimum silence length for splitting')
        split_parser.add_argument('--silence_thresh', type=int, default=1000, help='Silence threshold')
        split_parser.add_argument('--keep_silence', type=int, default=100, help='Amount of silence to keep')

        # Loop audio command
        loop_parser = subparsers.add_parser('loop_audio', help='Loop the current audio data')
        loop_parser.add_argument('--num_loops', type=int, default=10, help='Number of times to loop')

        # Cut selection command
        cut_parser = subparsers.add_parser('cut_selection', help='Cut a selection from the audio data')
        cut_parser.add_argument('start_time', type=float, help='Start time of the selection in seconds')
        cut_parser.add_argument('end_time', type=float, help='End time of the selection in seconds')

        args = parser.parse_args()

        if args.command in self.commands:
            self.commands[args.command].execute(self.processor, args)
        else:
            print(f"Unknown command: {args.command}")