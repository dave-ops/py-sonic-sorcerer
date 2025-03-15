import argparse
import wave
import struct
import os
import time
import numpy as np
from typing import List, Tuple

class AudioProcessor:
    def __init__(self):
        self.sample_rate: int = 0
        self.sample_width: int = 0
        self.audio_data: np.ndarray = np.array([])

    def load_wav(self, input_file: str) -> None:
        """Load a WAV file into memory."""
        with wave.open(input_file, 'rb') as wav_file:
            self.sample_rate = wav_file.getframerate()
            self.sample_width = wav_file.getsampwidth()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()

            raw_data = wav_file.readframes(n_frames)
            self.audio_data = np.frombuffer(raw_data, dtype=np.int16 if self.sample_width == 2 else np.int8)

            if n_channels > 1:
                self.audio_data = self.audio_data.reshape(-1, n_channels).mean(axis=1)

    def save_wav(self, output_file: str) -> None:
        """Save the current audio data to a WAV file."""
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(self.audio_data.astype(np.int16 if self.sample_width == 2 else np.int8).tobytes())

    def convert_stereo_to_mono(self) -> None:
        """Convert stereo audio to mono in-place."""
        if self.audio_data.ndim == 2:
            self.audio_data = self.audio_data.mean(axis=1)

    def trim_silence(self, silence_thresh: int = 1000, keep_silence: int = 100) -> None:
        """Trim leading and trailing silence from the audio data."""
        ms_per_frame = 1000 / self.sample_rate
        keep_frames = int(keep_silence / ms_per_frame)

        start_idx = 0
        for i, sample in enumerate(self.audio_data):
            if abs(sample) >= silence_thresh:
                start_idx = max(0, i - keep_frames)
                break

        end_idx = len(self.audio_data) - 1
        for i in range(len(self.audio_data) - 1, -1, -1):
            if abs(self.audio_data[i]) >= silence_thresh:
                end_idx = min(len(self.audio_data) - 1, i + keep_frames)
                break

        self.audio_data = self.audio_data[start_idx:end_idx + 1]

    def split_on_silence(self, output_dir: str, min_silence_len: int = 1000, silence_thresh: int = 1000, keep_silence: int = 100) -> None:
        """Split audio data into chunks based on silent pauses."""
        ms_per_frame = 1000 / self.sample_rate
        silence_frames = int(min_silence_len / ms_per_frame)
        keep_frames = int(keep_silence / ms_per_frame)

        chunk_ends = []
        start_idx = 0
        step_size = silence_frames // 2

        i = 0
        while i < len(self.audio_data) - silence_frames:
            window = self.audio_data[i:i + silence_frames]
            is_silence = all(abs(sample) < silence_thresh for sample in window)

            if is_silence:
                chunk_duration_ms = (i - start_idx) * ms_per_frame
                if chunk_duration_ms >= min_silence_len:
                    chunk_ends.append(i - keep_frames)
                    start_idx = i + silence_frames + keep_frames
                    i = start_idx
                else:
                    i += step_size
            else:
                i += step_size

        if start_idx < len(self.audio_data):
            chunk_ends.append(len(self.audio_data))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, end_idx in enumerate(chunk_ends):
            start_idx = 0 if i == 0 else chunk_ends[i - 1] + keep_frames
            chunk_data = self.audio_data[start_idx:end_idx]

            chunk_samples = len(chunk_data)
            chunk_duration_ms = chunk_samples * ms_per_frame
            print(f"Chunk {i}: {chunk_samples} samples, {chunk_duration_ms/1000:.2f} seconds")

            output_file = os.path.join(output_dir, f"chunk_{i}.wav")
            with wave.open(output_file, 'wb') as out_wav:
                out_wav.setnchannels(1)
                out_wav.setsampwidth(self.sample_width)
                out_wav.setframerate(self.sample_rate)
                out_wav.writeframes(chunk_data.astype(np.int16 if self.sample_width == 2 else np.int8).tobytes())

        print(f"Split into {len(chunk_ends)} chunks.")

    def loop_audio(self, num_loops: int = 10) -> None:
        """Loop the current audio data a specified number of times."""
        self.audio_data = np.tile(self.audio_data, num_loops)

    def cut_selection(self, start_time: float, end_time: float) -> None:
        """Cut a selection from the audio data."""
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        if start_sample < 0 or end_sample > len(self.audio_data) or start_sample >= end_sample:
            raise ValueError("Invalid selection range")

        self.audio_data = np.concatenate((self.audio_data[:start_sample], self.audio_data[end_sample:]))

class Command:
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        raise NotImplementedError

class LoadCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.load_wav(args.input_file)

class SaveCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.save_wav(args.output_file)

class ConvertToMonoCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.convert_stereo_to_mono()

class TrimSilenceCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.trim_silence(args.silence_thresh, args.keep_silence)

class SplitOnSilenceCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.split_on_silence(args.output_dir, args.min_silence_len, args.silence_thresh, args.keep_silence)

class LoopAudioCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.loop_audio(args.num_loops)

class CutSelectionCommand(Command):
    def execute(self, processor: AudioProcessor, args: argparse.Namespace) -> None:
        processor.cut_selection(args.start_time, args.end_time)

class SoundEditorCLI:
    def __init__(self):
        self.processor = AudioProcessor()
        self.commands = {
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

if __name__ == "__main__":
    editor = SoundEditorCLI()
    editor.run()