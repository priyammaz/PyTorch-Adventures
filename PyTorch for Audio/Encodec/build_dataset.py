import argparse
import random
from pathlib import Path


DEFAULT_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac"
}

def find_audio_files(root_dir, extensions, recursive=True):
    if recursive:
        iterator = root_dir.rglob("*")
    else:
        iterator = root_dir.glob("*")

    for path in iterator:
        if path.is_file() and path.suffix.lower() in extensions:
            yield path.resolve()


def save_list(paths, output_file, root_dir=None, relative=False):
    with open(output_file, "w") as f:
        for path in paths:
            if relative and root_dir is not None:
                path = path.relative_to(root_dir)
            f.write(str(path) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find audio files and optionally create train/test split"
    )

    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--relative", action="store_true")

    # Split options
    parser.add_argument("--split", action="store_true",
                        help="Enable train/test split")

    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train split ratio (default: 0.9)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--train_file", type=Path, default="train.txt")
    parser.add_argument("--test_file", type=Path, default="test.txt")

    parser.add_argument("--output", type=Path,
                        help="Output file if not splitting")

    args = parser.parse_args()

    root_dir = args.input_dir
    extensions = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    recursive = not args.no_recursive

    if not root_dir.exists():
        raise FileNotFoundError(root_dir)

    print("Scanning for audio files...")

    files = list(find_audio_files(root_dir, extensions, recursive))

    print(f"Found {len(files)} audio files")

    if len(files) == 0:
        return

    random.seed(args.seed)
    random.shuffle(files)

    if args.split:

        train_size = int(len(files) * args.train_ratio)

        train_files = files[:train_size]
        test_files = files[train_size:]

        save_list(train_files, args.train_file, root_dir, args.relative)
        save_list(test_files, args.test_file, root_dir, args.relative)

        print(f"Train: {len(train_files)} -> {args.train_file}")
        print(f"Test:  {len(test_files)} -> {args.test_file}")

    else:

        if args.output is None:
            raise ValueError("Must provide --output if not using --split")

        save_list(files, args.output, root_dir, args.relative)

        print(f"Saved {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
