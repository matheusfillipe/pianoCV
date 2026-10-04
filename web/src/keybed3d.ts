import type { Object3D, PerspectiveCamera } from "three";
import { Raycaster, Vector3 } from "three";
import type { Point } from "./homography";

// The scene's keyboard units, measured once from a reference piano mesh: the playable keys run
// y 0.0476 to 0.610, x -1.627 to 1.431, z -16.399 to 8.431. Span 24.83 by depth 3.058 is aspect
// 8.12, against 8.15 for a real keybed, and the black keys top out at y 0.8902, x 0.4204.
export const KEY_TOP_Y = 0.61;
export const KEY_BOTTOM_Y = 0.0476;
export const BLACK_TOP_Y = 0.8902;
export const BACK_X = -1.627;
export const FRONT_X = 1.431;
export const SPAN_MIN_Z = -16.399;
export const SPAN_MAX_Z = 8.431;

// the model's own key count, 52 white keys (88 keys total, A0 to C8); a smaller board is rendered
// as a contiguous crop of this range, since real keys are the same size on any size instrument
export const MODEL_WHITE_KEYS = 52;

export const SPAN = SPAN_MAX_Z - SPAN_MIN_Z;
export const DEPTH = FRONT_X - BACK_X;

// a real white key is 23.5mm wide, so this converts any real-world millimetre measurement
// (case panel depth, cheek width, ...) into the model's own units
export const WHITE_KEY_WIDTH_MM = 23.5;
const MODEL_KEY_WIDTH = SPAN / MODEL_WHITE_KEYS;

export function mmToUnits(mm: number): number {
  return (mm / WHITE_KEY_WIDTH_MM) * MODEL_KEY_WIDTH;
}

export interface KeybedCrop {
  readonly minZ: number;
  readonly maxZ: number;
}

// startFactor 0 crops from the back-pitch end of the model, 1 from the front-pitch end, and
// anything between; a full-size board (whiteKeys === MODEL_WHITE_KEYS) always crops to itself
export function keybedCrop(whiteKeys: number, startFactor: number): KeybedCrop {
  const keyWidth = SPAN / MODEL_WHITE_KEYS;
  const maxStart = MODEL_WHITE_KEYS - whiteKeys;
  const clamped = Math.min(1, Math.max(0, startFactor));
  const minZ = SPAN_MIN_Z + Math.round(clamped * maxStart) * keyWidth;
  return { minZ, maxZ: minZ + whiteKeys * keyWidth };
}

// corner 0 to 1 runs along the back edge, 1 to 2 crosses the depth, matching the label convention
export function cornersFor(minZ: number, maxZ: number): Vector3[] {
  return [
    new Vector3(BACK_X, KEY_TOP_Y, minZ),
    new Vector3(BACK_X, KEY_TOP_Y, maxZ),
    new Vector3(FRONT_X, KEY_TOP_Y, maxZ),
    new Vector3(FRONT_X, KEY_TOP_Y, minZ),
  ];
}

export function centreFor(minZ: number, maxZ: number): Vector3 {
  return new Vector3((BACK_X + FRONT_X) / 2, KEY_TOP_Y, (minZ + maxZ) / 2);
}

export const KEYBED_CORNERS: Vector3[] = cornersFor(SPAN_MIN_Z, SPAN_MAX_Z);
export const KEYBED_CENTRE = centreFor(SPAN_MIN_Z, SPAN_MAX_Z);

const CASE_TOP_STANDOFF_MM = 40;
export const CASE_BODY_DEPTH_MM = 90;
const CASE_RAIL_HEIGHT_MM = 18;
const CASE_RAIL_DEPTH_MM = 30;

export interface CaseBox {
  readonly size: readonly [number, number, number];
  readonly center: readonly [number, number, number];
}

export interface CaseLayout {
  readonly backPanel: CaseBox;
  readonly cheekLow: CaseBox;
  readonly cheekHigh: CaseBox;
  readonly frontRail: CaseBox;
  readonly body: CaseBox;
  // the panel's front face, and the raised strip above the black keys real control panels put
  // their buttons and screen on
  readonly panelFaceX: number;
  readonly panelTopY: number;
  readonly panelInnerTopY: number;
}

// boxes for the case dressing around a (possibly cropped) key span: a back panel behind the
// black keys' back end, end cheeks at both ends, a front rail under the white keys' front edge,
// and a body underneath. Every box stops exactly at the key mesh bounds, so the case can never
// cover a key.
export function caseLayout(
  crop: KeybedCrop,
  backDepthMm: number,
  cheekWidthMm: number,
  topStandoffMm = CASE_TOP_STANDOFF_MM,
): CaseLayout {
  const backDepth = mmToUnits(backDepthMm);
  const cheekWidth = mmToUnits(cheekWidthMm);
  const topY = BLACK_TOP_Y + mmToUnits(topStandoffMm);
  const railHeight = mmToUnits(CASE_RAIL_HEIGHT_MM);
  const railDepth = mmToUnits(CASE_RAIL_DEPTH_MM);
  const bodyBottomY = KEY_BOTTOM_Y - mmToUnits(CASE_BODY_DEPTH_MM);
  const railTopY = KEY_BOTTOM_Y;
  const railBottomY = railTopY - railHeight;
  const backFaceX = BACK_X - backDepth;
  const centreY = (topY + bodyBottomY) / 2;
  const heightFull = topY - bodyBottomY;
  const centreZ = (crop.minZ + crop.maxZ) / 2;
  const spanZ = crop.maxZ - crop.minZ;
  const outerMinZ = crop.minZ - cheekWidth;
  const outerMaxZ = crop.maxZ + cheekWidth;
  const cheekDepth = FRONT_X - backFaceX;
  const cheekCentreX = (backFaceX + FRONT_X) / 2;

  return {
    backPanel: {
      size: [backDepth, heightFull, spanZ],
      center: [(backFaceX + BACK_X) / 2, centreY, centreZ],
    },
    cheekLow: {
      size: [cheekDepth, heightFull, cheekWidth],
      center: [cheekCentreX, centreY, crop.minZ - cheekWidth / 2],
    },
    cheekHigh: {
      size: [cheekDepth, heightFull, cheekWidth],
      center: [cheekCentreX, centreY, crop.maxZ + cheekWidth / 2],
    },
    frontRail: {
      size: [railDepth, railHeight, spanZ],
      center: [FRONT_X - railDepth / 2, (railTopY + railBottomY) / 2, centreZ],
    },
    body: {
      size: [cheekDepth, railBottomY - bodyBottomY, outerMaxZ - outerMinZ],
      center: [
        cheekCentreX,
        (railBottomY + bodyBottomY) / 2,
        (outerMinZ + outerMaxZ) / 2,
      ],
    },
    panelFaceX: BACK_X,
    panelTopY: topY,
    panelInnerTopY: BLACK_TOP_Y,
  };
}

export interface PanelDetail {
  readonly size: readonly [number, number, number];
  readonly center: readonly [number, number, number];
  readonly lighter: number;
}

const PANEL_DETAIL_MIN_COUNT = 3;
const PANEL_DETAIL_MAX_COUNT = 8;
const PANEL_DETAIL_MARGIN_MM = 6;

// a scatter of small lighter boxes on the raised part of the back panel, standing in for the
// buttons, knobs, screen and labels a real control panel carries
export function panelDetails(
  layout: CaseLayout,
  random: () => number,
): PanelDetail[] {
  const margin = mmToUnits(PANEL_DETAIL_MARGIN_MM);
  const zMin =
    layout.backPanel.center[2] - layout.backPanel.size[2] / 2 + margin;
  const zMax =
    layout.backPanel.center[2] + layout.backPanel.size[2] / 2 - margin;
  const yMin = layout.panelInnerTopY + margin;
  const yMax = layout.panelTopY - margin;
  if (zMax <= zMin || yMax <= yMin) {
    return [];
  }
  const count =
    PANEL_DETAIL_MIN_COUNT +
    Math.floor(
      random() * (PANEL_DETAIL_MAX_COUNT - PANEL_DETAIL_MIN_COUNT + 1),
    );
  const details: PanelDetail[] = [];
  for (let i = 0; i < count; i += 1) {
    const width = Math.min(mmToUnits(8 + random() * 60), zMax - zMin);
    const height = Math.min(mmToUnits(4 + random() * 18), yMax - yMin);
    details.push({
      size: [mmToUnits(3), height, width],
      center: [
        layout.panelFaceX - mmToUnits(1.5),
        yMin + random() * (yMax - yMin),
        zMin + random() * (zMax - zMin),
      ],
      lighter: 0.15 + random() * 0.5,
    });
  }
  return details;
}

const scratch = new Vector3();

export function projectCorners(
  camera: PerspectiveCamera,
  corners: readonly Vector3[] = KEYBED_CORNERS,
): Point[] {
  return corners.map((corner) => {
    scratch.copy(corner).project(camera);
    return { x: (scratch.x + 1) / 2, y: (1 - scratch.y) / 2 };
  });
}

const CORNER_RAY_EPSILON = 0.02;
const cornerRay = new Raycaster();
const cameraPosition = new Vector3();
const ahead = new Vector3();

// a corner is visible when it is inside the camera frustum and nothing in `occluders` lies
// between the camera and it; the keys are left out of the occluders so the surface the corner
// sits on never hides it
export function cornersVisible(
  camera: PerspectiveCamera,
  corners: readonly Vector3[],
  occluders: readonly Object3D[],
): boolean[] {
  camera.getWorldPosition(cameraPosition);
  const solid = occluders.filter((occluder) => occluder.visible);
  return corners.map((corner) => {
    ahead.copy(corner).project(camera);
    const inFrustum =
      ahead.z < 1 &&
      ahead.x >= -1 &&
      ahead.x <= 1 &&
      ahead.y >= -1 &&
      ahead.y <= 1;
    if (!inFrustum) {
      return false;
    }
    const distance = cameraPosition.distanceTo(corner);
    cornerRay.set(
      cameraPosition,
      ahead.copy(corner).sub(cameraPosition).normalize(),
    );
    cornerRay.far = distance - CORNER_RAY_EPSILON;
    return cornerRay.intersectObjects([...solid], false).length === 0;
  });
}

const GRID = 9;
const sampleA = new Vector3();
const sampleB = new Vector3();
const sample = new Vector3();

// most real frames cut the keybed at the image edge, so what matters is how much of it landed,
// not whether all four corners did
export function visibleFraction(
  camera: PerspectiveCamera,
  corners: readonly Vector3[] = KEYBED_CORNERS,
): number {
  let inside = 0;
  for (let i = 0; i < GRID; i += 1) {
    const u = i / (GRID - 1);
    sampleA.lerpVectors(corners[0], corners[1], u);
    sampleB.lerpVectors(corners[3], corners[2], u);
    for (let j = 0; j < GRID; j += 1) {
      sample.lerpVectors(sampleA, sampleB, j / (GRID - 1)).project(camera);
      if (
        sample.z < 1 &&
        sample.x >= -1 &&
        sample.x <= 1 &&
        sample.y >= -1 &&
        sample.y <= 1
      ) {
        inside += 1;
      }
    }
  }
  return inside / (GRID * GRID);
}
