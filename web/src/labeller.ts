import type { BoardRead } from "./board";
import type { Corners } from "./calibrate";
import { captureFrame, postToLab } from "./lab";
import { createStillness, type Stillness } from "./stillness";
import type { TrackerState } from "./tracker";

export const saveEveryMs = 3000;
export const savesPerSession = 300;
export const agreementFloor = 0.9;

type Held = Extract<TrackerState, { kind: "held" }>;
type Source = "hand" | "board";

export interface LabelSidecar {
  readonly kind: "auto";
  readonly startedAt: number;
  readonly durationMs: number;
  readonly corners: Corners;
  readonly imageWidth: number;
  readonly imageHeight: number;
  readonly mimeType: string;
  /** Shared by every frame saved from one held keyboard, so a training split never tears one
   * hold across train and validation. */
  readonly session: string;
  readonly source: Source;
  readonly agreement?: number;
}

export type Save = (frame: HTMLVideoElement, sidecar: LabelSidecar) => void;

export interface LabellerOptions {
  readonly save?: Save;
  readonly stillness?: Stillness;
  readonly newSession?: () => string;
}

export interface Labeller {
  readonly saved: () => number;
  readonly look: (
    frame: HTMLVideoElement,
    tracker: TrackerState,
    board: BoardRead | null,
    enabled: boolean,
    now: number,
  ) => void;
}

function resolveSource(
  held: Held,
  board: BoardRead | null,
): { source: Source; agreement?: number } | null {
  if (held.byHand) {
    return { source: "hand" };
  }
  if (
    board !== null &&
    board.kind === "read" &&
    board.agreement >= agreementFloor
  ) {
    return { source: "board", agreement: board.agreement };
  }
  return null;
}

const defaultSave: Save = (frame, sidecar) => {
  void captureFrame(frame).then((blob) => {
    if (blob === null) {
      return;
    }
    const name = `auto-${sidecar.session}-${sidecar.startedAt}`;
    void postToLab(`${name}.png`, blob)
      .then(() =>
        postToLab(`${name}.json`, JSON.stringify(sidecar), "application/json"),
      )
      .catch(() => {
        // a dropped auto-label is not worth interrupting capture over; the next held frame tries again
      });
  });
};

/** Turns ordinary use of a held keyboard into training frames: every frame the demo app drives
 * through here, saving stills whenever the tracker and the board reader agree the corners are
 * trustworthy, spaced out so a still camera does not produce a pile of near-duplicates. */
export function createLabeller(options: LabellerOptions = {}): Labeller {
  const stillness = options.stillness ?? createStillness();
  const save = options.save ?? defaultSave;
  const newSession = options.newSession ?? (() => crypto.randomUUID());

  let current: Held | null = null;
  let session = "";
  let lastSavedAt = -Infinity;
  let saved = 0;

  return {
    saved: () => saved,
    look: (frame, tracker, board, enabled, now) => {
      if (tracker.kind !== "held") {
        current = null;
        return;
      }
      if (tracker !== current) {
        current = tracker;
        session = newSession();
        stillness.forget();
        lastSavedAt = -Infinity;
      }
      if (!enabled || saved >= savesPerSession) {
        return;
      }
      const resolved = resolveSource(tracker, board);
      if (resolved === null) {
        return;
      }
      if (now - lastSavedAt < saveEveryMs) {
        return;
      }
      if (!stillness.changed(frame, tracker.quad)) {
        return;
      }
      lastSavedAt = now;
      saved += 1;
      const [q0, q1, q2, q3] = tracker.quad;
      const corners: Corners = [q0, q1, q2, q3];
      save(frame, {
        kind: "auto",
        startedAt: now,
        durationMs: 0,
        corners,
        imageWidth: frame.videoWidth,
        imageHeight: frame.videoHeight,
        mimeType: "image/png",
        session,
        source: resolved.source,
        agreement: resolved.agreement,
      });
    },
  };
}
