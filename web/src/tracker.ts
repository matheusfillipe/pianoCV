import type { Detector } from "./detector";
import type { Point } from "./homography";
import { lockKeybed } from "./lock";
import { applyCalibration, type Measurement, measureCorners } from "./measure";
import { createSteady, type Steady } from "./steady";
import { createStillness, type Stillness } from "./stillness";

/** Hunting runs the model often since nothing is held yet. Held only has to
 * ask whether the keyboard moved, so a glance runs far less often. */
export const searchEveryMs = 350;
export const glanceEveryMs = 500;
export const confirmEveryMs = 2500;
// the model still gets the last word this often, so a drift or a swapped keyboard is caught
export const trustStillnessForMs = 30000;

export const readsToHold = 4;
export const agreeWithin = 0.02;

export const missesBeforeLost = 3;
// hands cover the keys for as long as they are played, so a held keybed is given far more patience
export const missesWhileHeld = 20;

export const staysWithin = 0.12;

export type Measured = Extract<Measurement, { kind: "depth" | "focal" }>;

export type Progress = {
  readonly agreed: number;
  readonly reason: string;
};

/** A keybed the tracker is still confirming, one it holds fixed, or one it
 * lost and stopped looking for. Corners are never moved once held: a held
 * keybed is held or it is gone. */
export type TrackerState =
  | { readonly kind: "hunting"; readonly progress: Progress }
  | { readonly kind: "held"; readonly quad: Point[]; readonly byHand: boolean }
  | { readonly kind: "lost"; readonly reason: string };

export type Reading = {
  readonly latencyMs: number;
  readonly coverage: number;
  readonly confidence: number;
};

export type TrackerOptions = {
  /** Told when a held keybed stops being there, so whatever draws it can
   * stop. */
  readonly onLost?: () => void;
  /** Told what a hand-placed quad measured, so the caller can persist it. */
  readonly onMeasured?: (measurement: Measured) => void;
  readonly stillness?: Stillness;
};

export interface Tracker {
  readonly look: (frame: HTMLVideoElement, now: number) => Promise<void>;
  readonly state: () => TrackerState;
  readonly reading: () => Reading | null;
  /** Takes the corners a person dragged, which hold until release(). */
  readonly hold: (quad: readonly Point[]) => void;
  readonly release: () => void;
}

function farthestCorner(a: readonly Point[], b: readonly Point[]): number {
  let worst = 0;
  for (let index = 0; index < 4; index += 1) {
    worst = Math.max(
      worst,
      Math.hypot(a[index].x - b[index].x, a[index].y - b[index].y),
    );
  }
  return worst;
}

export function createTracker(
  detector: Detector,
  options: TrackerOptions = {},
): Tracker {
  const stillness = options.stillness ?? createStillness();
  const steady: Steady = createSteady();

  let state: TrackerState = {
    kind: "hunting",
    progress: { agreed: 0, reason: "finding the keybed" },
  };
  let reading: Reading | null = null;
  let agreeing: Point[] | null = null;
  let agreed = 0;
  let misses = 0;
  let looking = false;
  let lastAt = 0;
  let lastRanAt = 0;
  let huntReason = "finding the keybed";
  let size = { width: 0, height: 0 };

  const hunt = (): void => {
    state = { kind: "hunting", progress: { agreed, reason: huntReason } };
  };

  const forget = (): void => {
    agreed = 0;
    agreeing = null;
    misses = 0;
    steady.reset();
    stillness.forget();
  };

  /** The keyboard is not where it was. Nothing is detected again until
   * release() is called: a camera left pointing at an instrument does not
   * lose it by accident, so this is worth saying rather than papering over. */
  const lose = (): void => {
    const wasHeld = state.kind === "held";
    forget();
    if (wasHeld) {
      state = { kind: "lost", reason: huntReason };
      options.onLost?.();
      return;
    }
    hunt();
  };

  /** Only corners a person placed say anything about the keybed's depth or
   * the lens: measuring the fit's own output would feed its assumptions back
   * into itself. */
  const settle = (quad: readonly Point[], fromHand: boolean): void => {
    if (fromHand) {
      const measured = measureCorners(quad, size);
      if (measured.kind === "depth") {
        applyCalibration({ depthUnits: measured.units, focalFraction: null });
        options.onMeasured?.(measured);
      } else if (measured.kind === "focal") {
        applyCalibration({
          depthUnits: null,
          focalFraction: measured.fraction,
        });
        options.onMeasured?.(measured);
      }
    }
    state = { kind: "held", quad: [...quad], byHand: fromHand };
  };

  return {
    look: async (frame, now) => {
      if (state.kind === "lost") {
        return;
      }
      const holding = state.kind === "held" ? state.quad : null;
      const every = holding === null ? searchEveryMs : glanceEveryMs;
      if (looking || now - lastAt < every) {
        return;
      }
      lastAt = now;
      const held = holding !== null;
      const moved = stillness.changed(frame, holding);
      if (held && !moved && now - lastRanAt < trustStillnessForMs) {
        return;
      }
      if (held && moved && now - lastRanAt < confirmEveryMs) {
        return;
      }
      looking = true;
      lastRanAt = now;
      size = { width: frame.videoWidth, height: frame.videoHeight };
      try {
        const detection = await detector.detect(frame);
        reading = {
          latencyMs: detection.latencyMs,
          coverage: detection.coverage,
          confidence: detection.confidence,
        };
        const lock = lockKeybed(detection, size);

        if (!lock.held) {
          huntReason = lock.reason;
          misses += 1;
          const patience =
            state.kind === "held" ? missesWhileHeld : missesBeforeLost;
          if (misses < patience) {
            return;
          }
          lose();
          return;
        }

        misses = 0;
        // Corners are never touched once they are held. The only question a
        // read answers from here is whether the keyboard is still where it
        // was put.
        if (state.kind === "held") {
          if (farthestCorner(lock.quad, state.quad) > staysWithin) {
            misses = missesBeforeLost;
            huntReason = "keybed moved out of place";
            lose();
          }
          return;
        }

        const settled = steady.accept(lock.quad, detection.still);
        if (
          agreeing !== null &&
          farthestCorner(settled, agreeing) < agreeWithin
        ) {
          agreed = Math.min(readsToHold, agreed + 1);
        } else {
          agreed = 1;
        }
        agreeing = settled;
        huntReason = "reading the keybed";
        if (agreed >= readsToHold) {
          settle(settled, false);
          return;
        }
        hunt();
      } finally {
        looking = false;
      }
    },
    state: () => state,
    reading: () => reading,
    hold: (quad) => settle(quad, true),
    release: () => {
      huntReason = "finding the keybed";
      forget();
      hunt();
    },
  };
}
