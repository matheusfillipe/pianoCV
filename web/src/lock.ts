import type { Detection } from "./detector";
import { INPUT_SIZE } from "./detector";
import type { Point } from "./homography";
import { facing } from "./orient";
import { canonicalQuad, solvePose } from "./pose";
import { checkQuad } from "./quad";

/** A keybed the picture supports, or the reason it was refused. Nothing on
 * screen beats the most likely wrong thing, so a refusal draws nothing. */
export type Lock =
  | {
      readonly held: true;
      /** Corners in frame coordinates, 0 to 1, ordered so 0 to 1 runs along the
       * back edge and 3 to 2 along the edge the player stands at. */
      readonly quad: Point[];
      readonly inputQuad: Point[];
      readonly margin: number;
    }
  | { readonly held: false; readonly reason: string };

export const minConfidence = 0.6;
export const maxPoseResidual = 1;

/** The fit's corner order says nothing about which long edge is the back. The
 * picture does: the black keys are darker than the white key fronts, so a quad
 * whose back band is the lighter one is turned around. */
function turnedIfReversed(detection: Detection): {
  quad: Point[];
  margin: number;
  onKeybed: boolean;
  inputQuad: Point[];
} {
  const source = detection.inputQuad ?? detection.quad ?? [];
  const asDetected = facing(detection.gray, INPUT_SIZE, source);
  const reversed = asDetected.margin < 0;
  const turn = (quad: Point[]): Point[] =>
    reversed
      ? [quad[2], quad[3], quad[0], quad[1]].filter(
          (corner): corner is Point => corner !== undefined,
        )
      : quad;
  const facts = reversed
    ? facing(detection.gray, INPUT_SIZE, turn(source))
    : asDetected;
  return {
    quad: canonicalQuad(turn(canonicalQuad(detection.quad ?? []))),
    margin: facts.margin,
    onKeybed: facts.onKeybed,
    inputQuad: facts.quad,
  };
}

/** Everything the picture has to say before a keybed is drawn: the mask filled
 * it, the shape is a keybed, it stands as a rectangle in 3D, and the black keys
 * are there. */
export function lockKeybed(
  detection: Detection,
  frame: { width: number; height: number },
): Lock {
  if (detection.quad === null) {
    return { held: false, reason: "no keybed" };
  }
  const framed = turnedIfReversed(detection);
  const shape = checkQuad(framed.quad);
  if (!shape.usable) {
    return { held: false, reason: shape.reason };
  }
  const residual = solvePose(
    framed.quad.map((corner) => ({
      x: corner.x * frame.width,
      y: corner.y * frame.height,
    })),
    frame.width,
    frame.height,
  ).residual;
  if (residual > maxPoseResidual) {
    return {
      held: false,
      reason: `not a rectangle in 3d (${residual.toFixed(1)})`,
    };
  }
  if (detection.confidence < minConfidence) {
    return {
      held: false,
      reason: `mask ${(detection.confidence * 100).toFixed(0)}% inside, not a keybed`,
    };
  }
  if (!framed.onKeybed) {
    return { held: false, reason: "no black-key stripe, not a keybed" };
  }
  return {
    held: true,
    quad: framed.quad,
    inputQuad: framed.inputQuad,
    margin: framed.margin,
  };
}
