"""Maestro is a bunch of long audio files, we will cut them down here for encodec pretraining"""

import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def chunk_audio_file(input_path, output_dir, chunk_duration=15.0, keep_remainder=True):
    """Cut a single audio file into chunks and save them."""
    try:
        audio, sr = librosa.load(str(input_path), sr=None, mono=False)
    except Exception as e:
        print(f"  Skipping {input_path}: {e}")
        return 0

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    else:
        audio = audio.T

    chunk_samples = int(chunk_duration * sr)
    total_samples = len(audio)
    num_chunks = total_samples // chunk_samples

    if num_chunks == 0:
        print(f"Skipping {input_path.name}: shorter than {chunk_duration}s")
        return 0

    stem = input_path.stem
    saved = 0

    def write_chunk(chunk, index):
        nonlocal saved
        out_filename = f"{stem}_chunk{index:04d}.wav"
        out_path = output_dir / out_filename
        if out_path.exists():
            out_filename = f"{input_path.parent.name}_{stem}_chunk{index:04d}.wav"
            out_path = output_dir / out_filename
        sf.write(str(out_path), chunk, sr)
        saved += 1

    for i in range(num_chunks):
        write_chunk(audio[i * chunk_samples:(i + 1) * chunk_samples], i)

    if keep_remainder:
        remainder = audio[num_chunks * chunk_samples:]
        if len(remainder) > 0:
            write_chunk(remainder, num_chunks)

    print(f"  {input_path.name} => {saved} chunks")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Recursively chunk audio files into fixed-length segments."
    )
    parser.add_argument("input_dir", help="Input folder containing audio files")
    parser.add_argument("output_dir", help="Output folder to save chunks")
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Chunk duration in seconds (default: 15)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".aif"],
        help="Audio file extensions to process",
    )
    parser.add_argument(
        "--keep-remainder",
        action="store_true",
        help="Save the final partial chunk without padding",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.extensions
    }

    audio_files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if not audio_files:
        print(f"No audio files found in '{input_dir}' with extensions: {extensions}")
        return

    print(f"Found {len(audio_files)} audio file(s). Chunking into {args.duration}s segments...\n")

    total_chunks = 0
    for audio_file in sorted(audio_files):
        total_chunks += chunk_audio_file(
            audio_file, output_dir, args.duration, args.keep_remainder
        )

    print(f"\nDone. {total_chunks} chunks saved to '{output_dir}'.")


if __name__ == "__main__":
    main()