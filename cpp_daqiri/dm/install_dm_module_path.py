"""Expose this directory to the active DigitalMicrograph Python environment."""

import argparse
from pathlib import Path
import site
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        type=Path,
        help="override the destination site-packages directory (tests only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    module_directory = Path(__file__).resolve().parent
    if args.site_packages is not None:
        site_packages = args.site_packages.resolve()
    else:
        candidates = site.getsitepackages()
        if not candidates:
            raise SystemExit("Python did not report a site-packages directory")
        site_packages = Path(candidates[0])

    site_packages.mkdir(parents=True, exist_ok=True)
    path_file = site_packages / "stem_daqiri_dm.pth"
    try:
        path_file.write_text(str(module_directory) + "\n", encoding="utf-8")
    except PermissionError as error:
        raise SystemExit(
            "Cannot write {}. Run this command from an Administrator shell."
            .format(path_file)
        ) from error

    print("Python executable: {}".format(sys.executable))
    print("Python prefix: {}".format(sys.prefix))
    print("Wrote: {}".format(path_file))
    print("STEM DM module directory: {}".format(module_directory))
    print("Restart DigitalMicrograph before importing stem_stream_protocol.")


if __name__ == "__main__":
    main()

