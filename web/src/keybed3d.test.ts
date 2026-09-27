import { PerspectiveCamera } from "three";
import { describe, expect, it } from "vitest";
import {
  BACK_X,
  BLACK_TOP_Y,
  caseLayout,
  centreFor,
  cornersFor,
  FRONT_X,
  KEY_BOTTOM_Y,
  KEYBED_CENTRE,
  KEYBED_CORNERS,
  keybedCrop,
  MODEL_WHITE_KEYS,
  mmToUnits,
  panelDetails,
  projectCorners,
  SPAN,
  SPAN_MAX_Z,
  SPAN_MIN_Z,
  visibleFraction,
} from "./keybed3d";

describe("keybedCrop", () => {
  it("crops to the full model when asked for every white key", () => {
    const crop = keybedCrop(MODEL_WHITE_KEYS, 0.5);
    expect(crop.minZ).toBeCloseTo(SPAN_MIN_Z);
    expect(crop.maxZ).toBeCloseTo(SPAN_MAX_Z);
  });

  it("gives a smaller board exactly its own share of the model's span", () => {
    const whiteKeys = 36;
    const crop = keybedCrop(whiteKeys, 0);
    expect(crop.maxZ - crop.minZ).toBeCloseTo(
      (SPAN * whiteKeys) / MODEL_WHITE_KEYS,
    );
  });

  it("never crops outside the model's own bounds, at either end of the start factor", () => {
    for (const startFactor of [0, 0.3, 0.7, 1]) {
      const crop = keybedCrop(45, startFactor);
      expect(crop.minZ).toBeGreaterThanOrEqual(SPAN_MIN_Z - 1e-9);
      expect(crop.maxZ).toBeLessThanOrEqual(SPAN_MAX_Z + 1e-9);
    }
  });

  it("clamps an out-of-range start factor instead of cropping past the model", () => {
    const atZero = keybedCrop(36, -5);
    const atOne = keybedCrop(36, 5);
    expect(atZero.minZ).toBeCloseTo(SPAN_MIN_Z);
    expect(atOne.maxZ).toBeCloseTo(SPAN_MAX_Z);
  });
});

describe("cornersFor and centreFor", () => {
  it("reproduce the full-model constants when given the model's own bounds", () => {
    const corners = cornersFor(SPAN_MIN_Z, SPAN_MAX_Z);
    for (const [i, corner] of corners.entries()) {
      expect(corner.x).toBeCloseTo(KEYBED_CORNERS[i].x);
      expect(corner.y).toBeCloseTo(KEYBED_CORNERS[i].y);
      expect(corner.z).toBeCloseTo(KEYBED_CORNERS[i].z);
    }
    const centre = centreFor(SPAN_MIN_Z, SPAN_MAX_Z);
    expect(centre.x).toBeCloseTo(KEYBED_CENTRE.x);
    expect(centre.z).toBeCloseTo(KEYBED_CENTRE.z);
  });
});

describe("mmToUnits", () => {
  it("scales a real white key width to the model's own key width", () => {
    expect(mmToUnits(23.5)).toBeCloseTo(SPAN / MODEL_WHITE_KEYS);
  });

  it("is linear in the millimetre amount", () => {
    expect(mmToUnits(100)).toBeCloseTo(mmToUnits(50) * 2);
  });
});

describe("caseLayout", () => {
  const crop = { minZ: SPAN_MIN_Z + 5, maxZ: SPAN_MIN_Z + 20 };
  const layout = caseLayout(crop, 120, 40);

  it("stands the back panel behind the back edge, never crossing it", () => {
    const maxX = layout.backPanel.center[0] + layout.backPanel.size[0] / 2;
    expect(maxX).toBeCloseTo(BACK_X);
    expect(layout.backPanel.size[0]).toBeCloseTo(mmToUnits(120));
  });

  it("stands above the black keys' own top", () => {
    expect(layout.panelTopY).toBeGreaterThan(BLACK_TOP_Y);
  });

  it("sits the cheeks outside the crop, never inside it", () => {
    const lowMaxZ = layout.cheekLow.center[2] + layout.cheekLow.size[2] / 2;
    const highMinZ = layout.cheekHigh.center[2] - layout.cheekHigh.size[2] / 2;
    expect(lowMaxZ).toBeCloseTo(crop.minZ);
    expect(highMinZ).toBeCloseTo(crop.maxZ);
    expect(layout.cheekLow.size[2]).toBeCloseTo(mmToUnits(40));
  });

  it("keeps the front rail below the white keys' own bottom", () => {
    const railTopY = layout.frontRail.center[1] + layout.frontRail.size[1] / 2;
    expect(railTopY).toBeCloseTo(KEY_BOTTOM_Y);
    const railMaxX = layout.frontRail.center[0] + layout.frontRail.size[0] / 2;
    expect(railMaxX).toBeCloseTo(FRONT_X);
  });

  it("stacks the body underneath the front rail", () => {
    const bodyTopY = layout.body.center[1] + layout.body.size[1] / 2;
    const railBottomY =
      layout.frontRail.center[1] - layout.frontRail.size[1] / 2;
    expect(bodyTopY).toBeCloseTo(railBottomY);
  });
});

describe("panelDetails", () => {
  const layout = caseLayout(
    { minZ: SPAN_MIN_Z + 5, maxZ: SPAN_MIN_Z + 20 },
    120,
    40,
  );

  it("scatters between 3 and 8 details onto the raised part of the panel", () => {
    for (let i = 0; i < 20; i += 1) {
      const seed = i / 20;
      const details = panelDetails(layout, () => seed);
      expect(details.length).toBeGreaterThanOrEqual(3);
      expect(details.length).toBeLessThanOrEqual(8);
      for (const detail of details) {
        expect(detail.center[1]).toBeGreaterThanOrEqual(layout.panelInnerTopY);
        expect(detail.center[1]).toBeLessThanOrEqual(layout.panelTopY);
        expect(detail.lighter).toBeGreaterThanOrEqual(0.15);
        expect(detail.lighter).toBeLessThanOrEqual(0.65);
      }
    }
  });
});

describe("projectCorners and visibleFraction with a custom crop", () => {
  it("reports full visibility for a cropped board framed dead-on", () => {
    const crop = { minZ: SPAN_MIN_Z + 5, maxZ: SPAN_MIN_Z + 15 };
    const corners = cornersFor(crop.minZ, crop.maxZ);
    const centre = centreFor(crop.minZ, crop.maxZ);
    const camera = new PerspectiveCamera(45, 4 / 3, 0.1, 500);
    camera.position.set(centre.x + 30, centre.y + 20, centre.z);
    camera.up.set(0, 1, 0);
    camera.lookAt(centre);
    camera.updateMatrixWorld();

    expect(visibleFraction(camera, corners)).toBeCloseTo(1);
    const projected = projectCorners(camera, corners);
    for (const point of projected) {
      expect(point.x).toBeGreaterThan(0);
      expect(point.x).toBeLessThan(1);
      expect(point.y).toBeGreaterThan(0);
      expect(point.y).toBeLessThan(1);
    }
  });
});
