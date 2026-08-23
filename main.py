"""Command-line entry point for Twisty Puzzle AI Lab."""

import argparse

from configs import PROFILE_ALIASES, PROFILE_NAMES, build_frame_config
from ui.frame import Frame


CLI_PROFILE_NAMES = PROFILE_NAMES + tuple(PROFILE_ALIASES)


def build_default_frame_config():
    """Return the lightweight public configuration for compatibility."""
    return build_frame_config("public")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch Twisty Puzzle AI Lab with a named configuration profile."
    )
    parser.add_argument(
        "--profile",
        choices=CLI_PROFILE_NAMES,
        default="public",
        help="public (default) or experiment; test is an alias for experiment",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame = Frame(config=build_frame_config(args.profile))
    frame.pack()
    frame.mainloop()


if __name__ == "__main__":
    main()
