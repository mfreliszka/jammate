"""Command-line entry point (`uv run jammate`)."""

import argparse
import logging
from collections.abc import Sequence

from pydantic import ValidationError

from jammate.config import AudioConfig, JamMateConfig

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jammate",
        description="Start playing. Your band joins in.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the jam engine from the terminal without the GUI",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list available audio input and output devices and exit",
    )
    parser.add_argument(
        "--input",
        metavar="NAME",
        help="input device name or unique substring (default: system default input)",
    )
    parser.add_argument(
        "--output",
        metavar="NAME",
        help="output device name or unique substring (default: system default output)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        metavar="FRAMES",
        help=f"audio block size in frames (default: {AudioConfig().block_size_frames})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    return parser


def build_config(args: argparse.Namespace) -> JamMateConfig:
    """Build the validated configuration from defaults plus command-line overrides."""
    audio_overrides: dict[str, object] = {
        "input_device_name": args.input,
        "output_device_name": args.output,
    }
    if args.block_size is not None:
        # An explicitly requested block size must never fall back to a smaller one.
        audio_overrides["block_size_frames"] = args.block_size
        audio_overrides["fallback_block_size_frames"] = max(
            args.block_size, AudioConfig().fallback_block_size_frames
        )
    return JamMateConfig(audio=AudioConfig.model_validate(audio_overrides))


def configure_logging(level_name: str) -> None:
    logging.basicConfig(level=level_name, format=LOG_FORMAT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = build_config(args)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        parser.error(f"invalid configuration: {details}")

    logger.debug("Configuration: %s", config.model_dump())

    if args.list_devices:
        return _not_implemented_yet("Device listing", "Step 1.2")
    if args.headless:
        return _not_implemented_yet("The headless jam runner", "Step 1.4")
    return _not_implemented_yet("The GUI", "Step 4.4")


def _not_implemented_yet(feature: str, step: str) -> int:
    logger.error("%s is not implemented yet (see %s in implementation_journal.md).", feature, step)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
