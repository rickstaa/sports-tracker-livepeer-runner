#!/usr/bin/env python3
"""sports-tracker app: realtime player/ball tracking over trickle.

Receives a live video stream over trickle `in`/`out` channels, detects players
and the ball (RF-DETR), tracks players across frames with Roboflow's `supervision`
ByteTrack, annotates the signature sports overlay (ellipse under each player, a
marker over the ball, id labels, movement traces), and streams the result back.
`teams` mode color-codes the two sides by jersey color.

Livepeer integration (grep `# Livepeer:`):
  1. register_runner()          — announce the app to the orchestrator (startup)
  2. create_trickle_channels()  — open the session's trickle in/out channels
  3. registration.close()       — deregister (cleanup)

Media I/O over trickle uses MediaOutput (read frames) and MediaPublish (write frames).
/track and /update are ordinary HTTP handlers; being on the network doesn't change how you write them.

Default weights are generic COCO (person + sports ball) so it runs on any
footage; pass a fine-tuned checkpoint via --weights for player / ball /
referee / goalkeeper classes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import av
import cv2
import numpy as np
from aiohttp import web

from livepeer_gateway.live_runner import register_runner
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish

log = logging.getLogger("sports-tracker")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8989
DEFAULT_APP = "rickstaa/sports-tracker"
TRACKERS = ("BoTSORT", "ByteTrack", "OCSORT", "SORT")
_tracker_name = "BoTSORT"
DEFAULT_SIZE = "medium"
MODES = frozenset({"track", "teams"})
SIZES = ("nano", "small", "medium", "large")

_person_class = None  # resolved by name, indexing differs between checkpoints
_ball_class = None

_model = None
_sv = None  # the supervision module
_ellipse = None  # single-color player ellipse
_ellipse_teams = None  # two-color (team) player ellipse
_label = None
_triangle = None  # ball marker


def _load(
    size: str, weights: str | None, precision: str = "fp16", compile: bool = False
) -> None:
    global _model, _sv, _ellipse, _ellipse_teams, _label, _triangle
    global _person_class, _ball_class
    if _model is not None:
        return
    import numpy as np
    import rfdetr
    import supervision as sv
    import torch
    from rfdetr.assets.coco_classes import COCO_CLASSES

    # RFDETRNano / RFDETRSmall / RFDETRMedium / RFDETRLarge, all Apache-2.0.
    cls = getattr(rfdetr, f"RFDETR{size.capitalize()}")
    _model = cls(pretrain_weights=weights) if weights else cls()

    # fp16 is most of the speedup and costs nothing on a Tensor Core GPU, so it is
    # the default. torch.compile is opt-in: it adds a warmup of tens of seconds and
    # is the piece that fails on an unusual driver/triton combination.
    dtype = torch.float16 if precision == "fp16" else torch.float32
    _model.inference(compile=compile, batch_size=1, dtype=dtype, inplace=True)

    # torch.compile is lazy, so force the warmup HERE, before register_runner
    # announces the app ready. Otherwise the first paying session buys the compile.
    _model.predict(np.zeros((720, 1280, 3), dtype=np.uint8), threshold=0.5)
    log.info("warmed up: precision=%s compile=%s", precision, compile)
    # Resolve by name: COCO checkpoints and fine-tuned ones index differently.
    by_name = {v: k for k, v in COCO_CLASSES.items()}
    _person_class = by_name.get("person", 0)
    _ball_class = by_name.get("sports ball", 32)
    _sv = sv
    _ellipse = sv.EllipseAnnotator()
    _ellipse_teams = sv.EllipseAnnotator(
        color=sv.ColorPalette([sv.Color(56, 168, 255), sv.Color(255, 92, 56)])
    )
    _label = sv.LabelAnnotator(text_scale=0.5)
    _triangle = sv.TriangleAnnotator()
    log.info("loaded RF-DETR %s + supervision", size)


def _new_tracker():
    """A tracker from roboflow/trackers (Apache-2.0).

    supervision's own ByteTrack is deprecated and removed in v0.31. BoT-SORT is
    the better default here anyway: it compensates for camera motion, and
    broadcast sports footage pans constantly, which is exactly when plain
    ByteTrack starts dropping ids.
    """
    import trackers

    cls = getattr(trackers, f"{_tracker_name}Tracker")
    return cls()


def _team_ids(img: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Cluster players into two teams by mean torso color (BGR, k-means=2)."""
    if len(xyxy) < 2:
        return np.zeros(len(xyxy), dtype=int)
    feats = []
    for x1, y1, x2, y2 in xyxy.astype(int):
        crop = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        h = crop.shape[0]
        torso = crop[h // 4 : h // 2, :] if h >= 4 else crop
        feats.append(torso.reshape(-1, 3).mean(axis=0) if torso.size else np.zeros(3))
    feats = np.asarray(feats, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(feats, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return labels.flatten()


@dataclass
class ModeState:
    mode: str = "track"


@dataclass
class TrackSession:
    session_id: str
    in_url: str
    out_url: str
    mode: ModeState
    output: MediaOutput
    publisher: MediaPublish
    tracker: Any = None
    trace: Any = None

    def to_json(self) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "in": self.in_url,
            "out": self.out_url,
            "mode": self.mode.mode,
        }


state: TrackSession | None = None


async def _close_pipeline() -> None:
    global state
    if state is None:
        return
    current = state
    state = None
    with suppress(Exception):
        await current.publisher.close()
    with suppress(Exception):
        await current.output.close()


def _annotate(img: np.ndarray, session: TrackSession) -> np.ndarray:
    # RF-DETR takes RGB and returns supervision Detections directly.
    det = _model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=0.5)
    players = det[det.class_id == _person_class]
    ball = det[det.class_id == _ball_class]
    # The frame is needed for camera-motion compensation, not just the boxes.
    players = session.tracker.update(players, img)
    # A tracker returns unconfirmed tracks as id -1; they become real after a few
    # frames. Drawing them labels half the pitch "#-1".
    if players.tracker_id is not None and len(players):
        players = players[players.tracker_id != -1]

    out = img.copy()
    if len(players):
        if session.mode.mode == "teams":
            players.class_id = _team_ids(img, players.xyxy)
            out = _ellipse_teams.annotate(out, players)
        else:
            out = _ellipse.annotate(out, players)
        labels = [f"#{tid}" for tid in players.tracker_id]
        out = _label.annotate(out, players, labels)
        out = session.trace.annotate(out, players)
    if len(ball):
        out = _triangle.annotate(out, ball)
    return out


def _transform_frame(decoded, session: TrackSession) -> av.VideoFrame | None:
    if decoded.kind != "video":
        return None
    frame = decoded.frame
    out_img = _annotate(frame.to_ndarray(format="bgr24"), session)
    out = av.VideoFrame.from_ndarray(out_img, format="bgr24")
    out.pts = frame.pts
    out.time_base = frame.time_base
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Runner sports player/ball tracker."
    )
    parser.add_argument("--orchestrator", default="https://localhost:8935")
    parser.add_argument("--orchSecret", default="abcdef")
    parser.add_argument("--runner-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    parser.add_argument("--app", default=DEFAULT_APP, help="App id to register.")
    parser.add_argument(
        "--price",
        type=float,
        default=0.0,
        help="USD per hour, metered per second. 0 registers free (offchain).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        choices=SIZES,
        help="RF-DETR variant. Bigger is more accurate and slower.",
    )
    parser.add_argument(
        "--tracker",
        default="BoTSORT",
        choices=TRACKERS,
        help="Tracking algorithm. BoT-SORT compensates for camera motion.",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "fp32"),
        help="fp16 is several times faster on Tensor Cores; fp32 to debug.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model. Faster, but adds a startup warmup.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Fine-tuned checkpoint. Omit for the COCO pretrained weights.",
    )
    return parser.parse_args()


def _session_id(request: web.Request) -> str:
    session_id = request.headers.get("Livepeer-Session-Id", "").strip()
    if not session_id:
        raise web.HTTPBadRequest(text="missing Livepeer-Session-Id header")
    return session_id


def _parse_mode(payload: dict[str, Any]) -> ModeState:
    mode = str(payload.get("mode", "track")).strip().lower()
    if mode not in MODES:
        raise web.HTTPBadRequest(text=f"mode must be one of {sorted(MODES)}")
    return ModeState(mode=mode)


async def _handle_track(request: web.Request) -> web.Response:
    global state
    session_id = _session_id(request)
    if state is not None:
        if state.session_id != session_id:
            raise web.HTTPConflict(
                text="sports-tracker runner already has an active session"
            )
        return web.json_response(state.to_json())

    # Pass the request so the SDK opens channels using the orchestrator's
    # Session-Control header, whose URLs are reachable from the runner's network.
    channels = await request.app["registration"].create_trickle_channels(  # Livepeer: 2
        request,
        [
            {"name": "in", "mime_type": "video/mp2t"},
            {"name": "out", "mime_type": "video/mp2t"},
        ],
    )
    by_name = {channel["name"]: channel for channel in channels}
    if "in" not in by_name or "out" not in by_name:
        raise web.HTTPInternalServerError(
            text="orchestrator did not return in/out channels"
        )

    mode = _parse_mode(json.loads(await request.read() or "{}"))
    publisher = MediaPublish(by_name["out"]["internal_url"])
    session = TrackSession(
        session_id=session_id,
        in_url=by_name["in"]["url"],
        out_url=by_name["out"]["url"],
        mode=mode,
        output=None,  # set below
        publisher=publisher,
        tracker=_new_tracker(),
        trace=_sv.TraceAnnotator(),
    )

    async def _on_frame(decoded) -> None:
        frame = _transform_frame(decoded, session)
        if frame is not None:
            await publisher.write_frame(frame)

    session.output = MediaOutput(by_name["in"]["internal_url"], on_frame=_on_frame)
    state = session
    for task in session.output.callback_tasks():
        task.add_done_callback(lambda _task: asyncio.create_task(_close_pipeline()))
    log.info("started sports-tracker session %s mode=%s", session_id, mode.mode)
    return web.json_response(state.to_json())


async def _handle_update(request: web.Request) -> web.Response:
    session_id = _session_id(request)
    if state is None:
        raise web.HTTPNotFound(text="sports-tracker session not started")
    if state.session_id != session_id:
        raise web.HTTPConflict(
            text="sports-tracker runner has a different active session"
        )
    state.mode.mode = _parse_mode(json.loads(await request.read() or "{}")).mode
    return web.json_response(state.to_json())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()
    global _tracker_name
    _tracker_name = args.tracker
    # Loads, optimizes and warms up before the app registers as ready.
    _load(args.size, args.weights, args.precision, args.compile)

    async def _on_startup(app: web.Application) -> None:
        app["registration"] = await register_runner(  # Livepeer: 1
            args.orchestrator,
            secret=args.orchSecret,
            runner_url=args.runner_url,
            app=args.app,
            price=args.price,  # USD per hour, metered per second while held
            mode="persistent",  # realtime trickle streaming is a held-open session
        )
        log.info(
            "registered runner_id=%s orchestrator=%s",
            app["registration"].runner_id,
            app["registration"].orchestrator_url,
        )

    async def _on_cleanup(app: web.Application) -> None:
        await _close_pipeline()
        with suppress(Exception):
            await app["registration"].close()  # Livepeer: 3

    app = web.Application()
    app.router.add_post("/track", _handle_track)
    app.router.add_post("/update", _handle_update)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    web.run_app(app, host=args.host, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
