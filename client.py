#!/usr/bin/env python3
"""sports-tracker client: reserve a session, stream video through the runner, settle up.

Publishes video frames into the runner's trickle `in` channel and reads the
annotated frames (player/ball tracking overlay) back from `out`. Input/output
can be files or stdin/stdout pipes, so you can chain `ffmpeg -> client -> ffplay`.

Livepeer integration (grep `# Livepeer:`):
  1. reserve_session()        — discover the runner, reserve a session
  2. MediaPublish/MediaOutput — publish frames to `in`, read tracked frames from `out`
  3. stop_runner_session()    — end the session (settles payment on-chain)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from contextlib import nullcontext, suppress
from pathlib import Path

import av

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.http import post_json
from livepeer_gateway.live_runner import stop_runner_session
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish
from livepeer_gateway.selection import reserve_session

DEFAULT_DISCOVERY = "https://localhost:8935/discovery"
APP_ID = "rickstaa/sports-tracker"  # keep in step with runner.py
DEFAULT_OUTPUT = "sports-tracker-out.ts"
MODES = ("track", "teams")

log = logging.getLogger("sports-tracker-client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the proxied sports-tracker Live Runner demo."
    )
    parser.add_argument(
        "input",
        help="input video file, or - to read an MPEG-TS stream from stdin (e.g. piped from ffmpeg)",
    )
    parser.add_argument(
        "--signer", default="", help="Remote signer base URL (on-chain/paid path)."
    )
    parser.add_argument("--discovery", default=DEFAULT_DISCOVERY)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="output file for the tracked stream, or - for stdout (e.g. piped to ffplay)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="track",
        help="track (single color) or teams (color-code by jersey).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many input video frames (0 = full file).",
    )
    return parser.parse_args()


def _channel_url(response: dict[str, object], name: str) -> str:
    url = response.get(name)
    if not isinstance(url, str) or not url:
        raise LivepeerGatewayError(f"track response missing {name!r} url")
    return url


async def _publish_video(
    input_source: str, publish_url: str, *, max_frames: int = 0
) -> None:
    # "-" = a live MPEG-TS stream on stdin; read it via libav's "pipe:0" rather than
    # sys.stdin.buffer, whose read() blocks for a full buffer and stalls until EOF.
    live = input_source == "-"
    input_ = av.open("pipe:0", format="mpegts") if live else av.open(input_source)
    try:
        if not input_.streams.video:
            raise LivepeerGatewayError(
                f"No video stream found in input: {input_source}"
            )
        publisher = MediaPublish(publish_url)  # Livepeer: 2 (publish frames)
        prev_pts_time: float | None = None
        prev_wall: float | None = None
        try:
            for index, frame in enumerate(input_.decode(video=0), start=1):
                if max_frames > 0 and index > max_frames:
                    break
                current_pts_time = None
                if frame.pts is not None and frame.time_base is not None:
                    current_pts_time = float(frame.pts * frame.time_base)

                # Pace files to realtime (live self-paces, so sleep_s=0). sleep(0) still
                # yields, so async POSTs/reads aren't starved by the blocking decode.
                sleep_s = 0.0
                if (
                    not live
                    and prev_pts_time is not None
                    and prev_wall is not None
                    and current_pts_time is not None
                ):
                    sleep_s = max(
                        0.0,
                        (current_pts_time - prev_pts_time)
                        - (time.monotonic() - prev_wall),
                    )
                if current_pts_time is not None:
                    prev_pts_time = current_pts_time
                    prev_wall = time.monotonic()

                await publisher.write_frame(frame)
                await asyncio.sleep(sleep_s)
        finally:
            await publisher.close()
    finally:
        input_.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    output_stdout = args.output.strip() == "-"
    output_path = None if output_stdout else Path(args.output).expanduser()
    input_source = args.input.strip()
    if input_source != "-":
        input_path = Path(input_source).expanduser()
        if not input_path.exists():
            raise SystemExit(f"input file does not exist: {input_path}")
        input_source = str(input_path)

    session = None
    try:
        session = await reserve_session(  # Livepeer: 1
            discovery_url=args.discovery,  # omit if the signer does discovery itself
            app=APP_ID,
            signer_url=args.signer.strip() or None,
        )
        log.info("session_id=%s app_url=%s", session.session_id, session.app_url)

        resp = await post_json(
            f"{session.app_url.rstrip('/')}/track", {"mode": args.mode}
        )
        in_url = _channel_url(resp, "in")
        out_url = _channel_url(resp, "out")
        log.info("in=%s out=%s mode=%s", in_url, out_url, args.mode)

        with (
            nullcontext(sys.stdout.buffer) if output_stdout else output_path.open("wb")
        ) as fh:

            def _write_chunk(chunk: bytes) -> None:
                fh.write(chunk)
                if output_stdout:
                    fh.flush()

            async with MediaOutput(
                out_url, on_bytes=_write_chunk
            ):  # Livepeer: 2 (read tracked frames)
                await _publish_video(
                    input_source, in_url, max_frames=max(0, args.max_frames)
                )
                log.info("publish complete; waiting for output to drain...")
            fh.flush()
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        if session is not None:
            with suppress(Exception):
                await stop_runner_session(session)  # Livepeer: 3


if __name__ == "__main__":
    asyncio.run(main())
