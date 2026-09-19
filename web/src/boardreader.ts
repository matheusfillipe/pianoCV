import { type BoardRead, type Picture, readBoard } from "./board";
import type { Board, KeybedSpace, PitchRange, Size } from "./keyspace";
import { boardOf } from "./keyspace";

/** How the keys are read: a few tries a second and a half apart, keeping the
 * best of them, and then never again. A try that explains the picture this
 * well ends it early. */
const everyMs = 1500;
const tries = 6;
const enough = 0.9;

/** How wide the picture the board is read from; colour across the keys needs
 * nothing like the detail the frame carries at full size. */
const readWidth = 640;

/** How long after a corner stops moving the keys are read again, so dragging
 * a handle across the keybed is one read rather than forty. */
const settleAfterMs = 500;

export type Capture = (frame: CanvasImageSource, size: Size) => Picture | null;

function canvasCapture(): Capture {
  let sheet: HTMLCanvasElement | null = null;
  return (frame, size) => {
    sheet ??= document.createElement("canvas");
    const scale = Math.min(1, readWidth / size.width);
    sheet.width = Math.round(size.width * scale);
    sheet.height = Math.round(size.height * scale);
    const context = sheet.getContext("2d", { willReadFrequently: true });
    if (context === null) {
      return null;
    }
    context.drawImage(frame, 0, 0, sheet.width, sheet.height);
    const pixels = context.getImageData(0, 0, sheet.width, sheet.height);
    return {
      width: pixels.width,
      height: pixels.height,
      data: pixels.data,
      scale,
    };
  };
}

export type BoardReader = {
  /** The board found so far, or null while the picture has not answered. */
  readonly board: () => Board | null;
  /** What the last try found, including why it was unsure, or null before
   * any try has run. */
  readonly last: () => BoardRead | null;
  /** Reads the keys again where a read is still owed. Cheap to call every
   * frame: it keeps its own clock and its own budget. A caller with a pinned
   * range skips the picture entirely and takes it as the board. */
  readonly look: (
    space: KeybedSpace,
    frame: CanvasImageSource,
    size: Size,
    now: number,
    pinned?: PitchRange | null,
  ) => void;
  /** The corners changed, so the last answer was about a different keybed.
   * The keys are read again once they stop moving. */
  readonly moved: (now: number) => void;
  /** A note the player sounded. Colour says how many keys there are and which
   * note the board starts on inside an octave, never which octave, so notes
   * played narrow which octave the next read may answer. */
  readonly played: (pitch: number) => void;
};

export function createBoardReader(
  capture: Capture = canvasCapture(),
): BoardReader {
  let at = 0;
  let taken = 0;
  let agreement = 0;
  let board: Board | null = null;
  let last: BoardRead | null = null;
  let sounded: PitchRange | null = null;
  let dueAt: number | null = null;

  return {
    board: () => board,
    last: () => last,
    look: (space, frame, size, now, pinned = null) => {
      if (dueAt !== null && now < dueAt) {
        return;
      }
      dueAt = null;
      const settled = taken >= tries || agreement >= enough;
      if (settled || now - at < everyMs) {
        return;
      }
      at = now;
      taken += 1;
      if (pinned !== null) {
        board = boardOf(pinned);
        return;
      }
      const picture = capture(frame, size);
      if (picture === null) {
        return;
      }
      const found = readBoard(space, picture, sounded);
      last = found;
      if (found.kind === "read" && found.agreement > agreement) {
        board = found.board;
        agreement = found.agreement;
      }
    },
    moved: (now) => {
      taken = 0;
      agreement = 0;
      board = null;
      last = null;
      dueAt = now + settleAfterMs;
    },
    played: (pitch) => {
      if (
        sounded !== null &&
        pitch >= sounded.lowest &&
        pitch <= sounded.highest
      ) {
        return;
      }
      sounded =
        sounded === null
          ? { lowest: pitch, highest: pitch }
          : {
              lowest: Math.min(sounded.lowest, pitch),
              highest: Math.max(sounded.highest, pitch),
            };
    },
  };
}
