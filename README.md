# Sonic Sorcer CLI

A command-line sound mixer for editing and manipulating audio files.

## Overview

Sonic Sorcer CLI is a Python-based tool designed to provide a simple yet powerful interface for audio editing and manipulation. It allows users to perform various audio processing tasks directly from the command line, making it ideal for quick edits, batch processing, and integration into automated workflows.

## Features

- Load and save WAV files
- Convert stereo audio to mono
- Trim leading and trailing silence
- Split audio into chunks based on silent pauses
- Loop audio a specified number of times
- Cut specific selections from audio

## Installation

1. Ensure you have Python 3.8 or later installed.
2. Clone the repository
3. Navigate to the project directory:
```
   cd sonic-sorcer-cli
```
4. Install the required dependencies:
```
   pip install numpy
```

## Usage
Sonic Sorcer CLI uses a command-based interface. Here are some example commands:

```bash
# load a wav file
python main.py load input.wav

# convert stereo to mono
python main.py convert_to_mono

# trim silence
python main.py trim_silence --silence_thresh 1000 --keep_silence 50

# split on silence
python main.py split_on_silence output_dir --min_silence_len 1000 --silence_thresh 1000 --keep_silence 100

# loop audio
python main.py loop_audio --num_loops 5

# cut a selection
python main.py cut_selection 1.5 3.0

# save the result
python main.py save output.wav
```

For a full list of available commands and options, run:
```bash
python main.py --help
```