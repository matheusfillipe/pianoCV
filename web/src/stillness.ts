import type { Point } from "./homography";

const ACROSS = 48;
const DOWN = 27;
// sensor noise in a dim room drifts a level or two; a camera nudge drifts every sample
const MOVED_BY = 6;

export interface Stillness {
  readonly changed: (
    frame: CanvasImageSource,
    keybed: readonly Point[] | null,
  ) => boolean;
  readonly forget: () => void;
}

// unit-agnostic point-in-polygon test, shared with follow.ts which samples in pixel space
// rather than this module's frame fractions
export function pointInQuad(quad: readonly Point[], at: Point): boolean {
  let within = false;
  for (let index = 0; index < quad.length; index += 1) {
    const one = quad[index];
    const other = quad[(index + quad.length - 1) % quad.length];
    if (
      one.y > at.y !== other.y > at.y &&
      at.x < ((other.x - one.x) * (at.y - one.y)) / (other.y - one.y) + one.x
    ) {
      within = !within;
    }
  }
  return within;
}

/** Whether the camera is looking at the same scene it was. A held keybed does
 * not move while it is played, so the model only has to run again once the
 * picture around it does. */
export function createStillness(): Stillness {
  const sheet = document.createElement("canvas");
  sheet.width = ACROSS;
  sheet.height = DOWN;
  let last: Uint8ClampedArray | null = null;

  return {
    changed: (frame, keybed) => {
      const context = sheet.getContext("2d", { willReadFrequently: true });
      if (context === null) {
        return true;
      }
      context.drawImage(frame, 0, 0, ACROSS, DOWN);
      const now = context.getImageData(0, 0, ACROSS, DOWN).data;
      const before = last;
      last = now;
      if (before === null || before.length !== now.length) {
        return true;
      }
      let drift = 0;
      let counted = 0;
      for (let y = 0; y < DOWN; y += 1) {
        for (let x = 0; x < ACROSS; x += 1) {
          const at = { x: (x + 0.5) / ACROSS, y: (y + 0.5) / DOWN };
          if (keybed !== null && pointInQuad(keybed, at)) {
            continue;
          }
          const index = (y * ACROSS + x) * 4;
          drift += Math.abs(now[index] - before[index]);
          counted += 1;
        }
      }
      return counted === 0 || drift / counted > MOVED_BY;
    },
    forget: () => {
      last = null;
    },
  };
}
