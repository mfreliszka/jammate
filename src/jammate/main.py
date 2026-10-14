"""Command-line entry point (`uv run jammate`)."""

import argparse
import logging
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from jammate.audio.devices import (
    AudioDevice,
    DeviceKind,
    DeviceResolutionError,
    format_device_list,
    query_inventory,
    resolve_device,
    supports_settings,
)
from jammate.config import (
    INSTRUMENT_PITCH_RANGES_HZ,
    MODEL_SIZE_LABELS,
    AccompanimentConfig,
    AnalysisConfig,
    AudioConfig,
    JamMateConfig,
)
from jammate.headless import run_headless
from jammate.ui.main_window import run_gui

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


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
        "--input-channel",
        type=_positive_int,
        metavar="N",
        help="input channel carrying the instrument, 1 = first input (default: 1)",
    )
    parser.add_argument(
        "--monitor-gain",
        type=float,
        metavar="GAIN",
        help=(
            "linear gain of the bass in the output, 0 to 4; use 0 with an interface's direct "
            f"monitoring (default: {AudioConfig().monitor_gain})"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        metavar="FRAMES",
        help=f"audio block size in frames (default: {AudioConfig().block_size_frames})",
    )
    parser.add_argument(
        "--instrument",
        choices=sorted(INSTRUMENT_PITCH_RANGES_HZ),
        help=(
            "instrument plugged in; selects the pitch range "
            f"(default: {AnalysisConfig().instrument})"
        ),
    )
    parser.add_argument(
        "--bpm",
        type=float,
        metavar="BPM",
        help="lock tempo to BPM manually, bypassing tempo estimation (§58)",
    )
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SIZE_LABELS),
        help=(
            "AI accompaniment model quality: mrt2_small (low quality/fast) or mrt2_base "
            f"(high quality) (default: {AccompanimentConfig().model_size})"
        ),
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
    if args.input_channel is not None:
        # Channels are numbered from 1 on the command line, as on the interface.
        audio_overrides["input_channel_index"] = args.input_channel - 1
    if args.monitor_gain is not None:
        audio_overrides["monitor_gain"] = args.monitor_gain
    if args.block_size is not None:
        # An explicitly requested block size must never fall back to a smaller one.
        audio_overrides["block_size_frames"] = args.block_size
        audio_overrides["fallback_block_size_frames"] = max(
            args.block_size, AudioConfig().fallback_block_size_frames
        )
    analysis_overrides: dict[str, object] = {}
    if args.instrument is not None:
        analysis_overrides["instrument"] = args.instrument
    if args.bpm is not None:
        analysis_overrides["tempo"] = {"manual_bpm": args.bpm}
    ai_overrides: dict[str, object] = {}
    if getattr(args, "model_size", None) is not None:
        ai_overrides["model_size"] = args.model_size
    return JamMateConfig(
        audio=AudioConfig.model_validate(audio_overrides),
        analysis=AnalysisConfig.model_validate(analysis_overrides),
        ai=AccompanimentConfig.model_validate(ai_overrides),
    )


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
        return list_devices(config)
    if args.headless:
        return run_headless(config)
    return run_gui(config)


def list_devices(config: JamMateConfig) -> int:
    """Print the device table and the devices that --input/--output would select."""
    inventory = query_inventory()
    sample_rate_hz = config.audio.sample_rate_hz

    def is_rate_supported(device: AudioDevice, kind: DeviceKind) -> bool:
        channels = 1 if kind == "input" else min(2, device.max_output_channels)
        return supports_settings(device, kind, sample_rate_hz, channels)

    print(format_device_list(inventory, sample_rate_hz, is_rate_supported))
    print()

    exit_code = 0
    selections: tuple[tuple[DeviceKind, str | None], ...] = (
        ("input", config.audio.input_device_name),
        ("output", config.audio.output_device_name),
    )
    for kind, query in selections:
        try:
            device = resolve_device(inventory, kind, query)
        except DeviceResolutionError as exc:
            print(f"Selected {kind}: error: {exc}", file=sys.stderr)
            exit_code = 1
        else:
            print(f"Selected {kind}: {device.name}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
