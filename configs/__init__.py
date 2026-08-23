"""Named startup configurations for Twisty Puzzle AI Lab."""

from .profiles import build_experiment_frame_config, build_public_frame_config


PROFILE_NAMES = ("public", "experiment")
PROFILE_ALIASES = {"test": "experiment"}


def build_frame_config(profile="public"):
    """Build a GUI configuration from a public profile name or alias."""
    normalized_profile = PROFILE_ALIASES.get(profile, profile)
    builders = {
        "public": build_public_frame_config,
        "experiment": build_experiment_frame_config,
    }
    try:
        builder = builders[normalized_profile]
    except KeyError as error:
        choices = ", ".join(PROFILE_NAMES)
        raise ValueError(
            f"Unknown profile: {profile!r}. Choose one of: {choices}."
        ) from error
    return builder()


__all__ = [
    "PROFILE_ALIASES",
    "PROFILE_NAMES",
    "build_experiment_frame_config",
    "build_frame_config",
    "build_public_frame_config",
]
