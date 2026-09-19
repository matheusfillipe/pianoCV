"""Extract labeled frames from recorded snapshots, clips, and gemini scenes."""

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from kvt.template import STRIP_HEIGHT, STRIP_WIDTH, strip_destination

_FRAMES_PER_CLIP = 40
_LABELS_NAME = "labels.json"
_GEMINI_SUFFIX = "-orig.png"
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECORDINGS_DIR = _REPO_ROOT / "data" / "recordings"
DEFAULT_FRAMES_DIR = _REPO_ROOT / "data" / "frames"
DEFAULT_GEMINI_DIR = _REPO_ROOT / "data" / "gemini"
DEFAULT_SYNTH_DIR = _REPO_ROOT / "data" / "synth"


@dataclass
class Recording:
    stem: str
    kind: str
    media_path: Path
    sidecar_path: Path


@dataclass
class Frame:
    image_path: Path
    corners_px: np.ndarray | None
    source_stem: str
    kind: str
    session: str | None = None


@dataclass
class Sidecar:
    kind: str
    corners: np.ndarray
    width: int
    height: int
    session: str | None = None


@dataclass
class FrameEntry:
    corners_px: np.ndarray | None
    source_stem: str
    kind: str
    session: str | None = None


@dataclass
class Labels:
    extracted: dict[str, int] = field(default_factory=dict)
    frames: dict[str, FrameEntry] = field(default_factory=dict)


def canonical_quad(quad: np.ndarray) -> np.ndarray:
    # every consumer maps the 52 white keys onto edge 0->1 and the keybed depth onto edge 1->2
    if _edge_length(quad, 1) >= _edge_length(quad, 3):
        return np.asarray(quad, dtype=np.float64)
    return np.roll(quad, -1, axis=0).astype(np.float64)


def orient_quad(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    # canonical_quad leaves a 180 degree ambiguity; the black keys settle it, they sit at the back
    ordered = canonical_quad(quad)
    flipped = np.roll(ordered, 2, axis=0)
    return (
        ordered
        if _back_darkness(image_bgr, ordered) < _back_darkness(image_bgr, flipped)
        else flipped
    )


def _back_darkness(image_bgr: np.ndarray, quad: np.ndarray) -> float:
    matrix = cv2.getPerspectiveTransform(
        quad.astype(np.float32), strip_destination(STRIP_WIDTH, STRIP_HEIGHT)
    )
    strip = cv2.warpPerspective(image_bgr, matrix, (STRIP_WIDTH, STRIP_HEIGHT))
    return float(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)[: STRIP_HEIGHT // 3].mean())


def _edge_length(quad: np.ndarray, index: int) -> float:
    return float(np.linalg.norm(quad[index] - quad[0]))


def align(quad: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The quad's corners in the truth's order, whatever order the fit produced them in.

    Canonical order picks the longer projected edge, and on a foreshortened keybed that is
    the wrong one, so a fit that is right on the pixels can be compared corner to wrong corner.
    """
    best = quad
    best_distance = float("inf")
    for candidate in (quad, quad[::-1]):
        for roll in range(4):
            rolled = np.roll(candidate, roll, axis=0)
            distance = float(np.linalg.norm(rolled - truth, axis=1).sum())
            if distance < best_distance:
                best, best_distance = rolled, distance
    return np.asarray(best, dtype=np.float64)


def scan_recordings(recordings_dir: Path) -> list[Recording]:
    recordings: list[Recording] = []
    for media_path in sorted(recordings_dir.iterdir()):
        suffix = media_path.suffix.lower()
        if suffix not in {".png", ".webm"}:
            continue
        sidecar_path = media_path.with_suffix(".json")
        if not sidecar_path.is_file():
            continue
        kind = "snap" if suffix == ".png" else "rec"
        recordings.append(
            Recording(
                stem=media_path.stem,
                kind=kind,
                media_path=media_path,
                sidecar_path=sidecar_path,
            )
        )
    return recordings


def load_synth(synth_dir: Path = DEFAULT_SYNTH_DIR) -> list[Frame]:
    # rendered frames carry projected corners, already in convention, so they are never re-oriented
    if not synth_dir.is_dir():
        return []
    frames: list[Frame] = []
    for image_path in sorted(synth_dir.glob("*.png")):
        sidecar_path = image_path.with_suffix(".json")
        if not sidecar_path.is_file():
            continue
        sidecar = parse_sidecar(sidecar_path, "synth")
        frames.append(
            Frame(
                image_path=image_path,
                corners_px=sidecar.corners
                * np.array([float(sidecar.width), float(sidecar.height)]),
                source_stem=image_path.stem,
                kind="synth",
            )
        )
    return frames


def scan_gemini(gemini_dir: Path) -> list[Recording]:
    if not gemini_dir.is_dir():
        return []
    recordings: list[Recording] = []
    for media_path in sorted(gemini_dir.glob(f"*{_GEMINI_SUFFIX}")):
        stem = media_path.name.removesuffix(_GEMINI_SUFFIX)
        sidecar_path = gemini_dir / f"{stem}.json"
        if not sidecar_path.is_file():
            continue
        recordings.append(
            Recording(
                stem=stem,
                kind="gemini",
                media_path=media_path,
                sidecar_path=sidecar_path,
            )
        )
    return recordings


def extract(recordings_dir: Path, frames_dir: Path, gemini_dir: Path | None = None) -> list[Frame]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_labels(frames_dir)
    recordings = scan_recordings(recordings_dir)
    if gemini_dir is not None:
        recordings += scan_gemini(gemini_dir)
    # the media may be deleted once extracted, the label may not: a sidecar still present
    # keeps its frames, a sidecar taken away takes them with it
    _forget_missing(frames_dir, labels, {path.stem for path in recordings_dir.glob("*.json")})
    for recording in recordings:
        if recording.stem in labels.extracted:
            continue
        try:
            sidecar = parse_sidecar(
                recording.sidecar_path, "gemini" if recording.kind == "gemini" else None
            )
            corners_px = sidecar.corners * np.array([float(sidecar.width), float(sidecar.height)])
            if recording.kind == "snap":
                _extract_snap(
                    recording, frames_dir, corners_px, labels, sidecar.kind, sidecar.session
                )
            elif recording.kind == "gemini":
                _extract_gemini(recording, frames_dir, corners_px, labels)
            else:
                _extract_clip(recording, frames_dir, corners_px, labels)
        except ValueError as error:
            warnings.warn(f"skipping recording {recording.media_path}: {error}", stacklevel=2)
            continue
        labels.extracted[recording.stem] = sum(
            1 for entry in labels.frames.values() if entry.source_stem == recording.stem
        )
    _write_labels(frames_dir, labels)
    return load_frames(frames_dir)


def _forget_missing(frames_dir: Path, labels: Labels, present: set[str]) -> None:
    # a label withdrawn from the recordings dir has to leave here too, or training keeps
    # reading a frame whose label was taken away for being wrong
    for name, entry in list(labels.frames.items()):
        if entry.kind != "gemini" and entry.source_stem not in present:
            del labels.frames[name]
            (frames_dir / name).unlink(missing_ok=True)
    for stem in list(labels.extracted):
        if stem not in present:
            del labels.extracted[stem]


def load_frames(frames_dir: Path) -> list[Frame]:
    labels = _load_labels(frames_dir)
    return [
        Frame(
            image_path=frames_dir / name,
            corners_px=entry.corners_px,
            source_stem=entry.source_stem,
            kind=entry.kind,
            session=entry.session,
        )
        for name, entry in sorted(labels.frames.items())
    ]


def _extract_snap(
    recording: Recording,
    frames_dir: Path,
    corners_px: np.ndarray,
    labels: Labels,
    kind: str,
    session: str | None,
) -> None:
    image = cv2.imread(str(recording.media_path))
    if image is None:
        raise ValueError(f"cannot read snapshot {recording.media_path}")
    name = f"{recording.stem}.png"
    shutil.copyfile(recording.media_path, frames_dir / name)
    labels.frames[name] = FrameEntry(
        corners_px=orient_quad(image, corners_px),
        source_stem=recording.stem,
        kind=kind,
        session=session,
    )


def _extract_gemini(
    recording: Recording,
    frames_dir: Path,
    corners_px: np.ndarray,
    labels: Labels,
) -> None:
    image = cv2.imread(str(recording.media_path))
    if image is None:
        raise ValueError(f"cannot read snapshot {recording.media_path}")
    name = f"{recording.stem}{_GEMINI_SUFFIX}"
    shutil.copyfile(recording.media_path, frames_dir / name)
    labels.frames[name] = FrameEntry(
        corners_px=orient_quad(image, corners_px),
        source_stem=recording.stem,
        kind="gemini",
    )


def _extract_clip(
    recording: Recording,
    frames_dir: Path,
    corners_px: np.ndarray,
    labels: Labels,
) -> None:
    capture = cv2.VideoCapture(str(recording.media_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot open clip {recording.media_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(total / _FRAMES_PER_CLIP)) if total > 0 else 1
    index = 0
    oriented: np.ndarray | None = None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if oriented is None:
                oriented = orient_quad(image, corners_px)
            if index % step == 0:
                name = f"{recording.stem}.{index:06d}.png"
                cv2.imwrite(str(frames_dir / name), image)
                labels.frames[name] = FrameEntry(
                    corners_px=oriented,
                    source_stem=recording.stem,
                    kind="rec",
                )
            index += 1
    finally:
        capture.release()


def parse_sidecar(path: Path, default_kind: str | None = None) -> Sidecar:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"sidecar {path} is not a json object")
    kind = data.get("kind", default_kind)
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    corners = data.get("corners")
    session = data.get("session")
    if not isinstance(kind, str) or not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"sidecar {path} has invalid metadata")
    if session is not None and not isinstance(session, str):
        raise ValueError(f"sidecar {path} has an invalid session")
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError(f"sidecar {path} must list exactly 4 corners")
    points = np.zeros((4, 2), dtype=np.float64)
    for i, corner in enumerate(corners):
        if not isinstance(corner, dict):
            raise ValueError(f"sidecar {path} corner {i} is not an object")
        x = corner.get("x")
        y = corner.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"sidecar {path} corner {i} has invalid coordinates")
        points[i] = (float(x), float(y))
    scale = np.array([float(width), float(height)])
    return Sidecar(
        kind=kind,
        corners=canonical_quad(points * scale) / scale,
        width=width,
        height=height,
        session=session,
    )


def _load_labels(frames_dir: Path) -> Labels:
    path = frames_dir / _LABELS_NAME
    if not path.is_file():
        return Labels()
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"labels file {path} is not a json object")
    labels = Labels()
    extracted = data.get("extracted")
    if isinstance(extracted, dict):
        for stem, count in extracted.items():
            if isinstance(stem, str) and isinstance(count, int):
                labels.extracted[stem] = count
    frames = data.get("frames")
    if isinstance(frames, dict):
        for name, entry in frames.items():
            parsed = _parse_frame_entry(entry)
            if parsed is not None and isinstance(name, str):
                labels.frames[name] = parsed
    return labels


def _parse_frame_entry(entry: object) -> FrameEntry | None:
    if not isinstance(entry, dict):
        return None
    source_stem = entry.get("source_stem")
    kind = entry.get("kind")
    if not isinstance(source_stem, str) or not isinstance(kind, str):
        return None
    session = entry.get("session")
    if session is not None and not isinstance(session, str):
        return None
    raw_corners = entry.get("corners_px")
    if raw_corners is None:
        return FrameEntry(corners_px=None, source_stem=source_stem, kind=kind, session=session)
    if not isinstance(raw_corners, list) or len(raw_corners) != 4:
        return None
    corners = np.zeros((4, 2), dtype=np.float64)
    for i, point in enumerate(raw_corners):
        if not isinstance(point, list) or len(point) != 2:
            return None
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        corners[i] = (float(x), float(y))
    return FrameEntry(
        corners_px=canonical_quad(corners), source_stem=source_stem, kind=kind, session=session
    )


def _write_labels(frames_dir: Path, labels: Labels) -> None:
    data = {
        "extracted": labels.extracted,
        "frames": {
            name: {
                "corners_px": None if entry.corners_px is None else entry.corners_px.tolist(),
                "source_stem": entry.source_stem,
                "kind": entry.kind,
                "session": entry.session,
            }
            for name, entry in sorted(labels.frames.items())
        },
    }
    (frames_dir / _LABELS_NAME).write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="extract labeled frames from recordings")
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--gemini-dir", type=Path, default=DEFAULT_GEMINI_DIR)
    args = parser.parse_args()
    frames = extract(args.recordings_dir, args.frames_dir, args.gemini_dir)
    print(f"{len(frames)} frames in {args.frames_dir}")


if __name__ == "__main__":
    main()
