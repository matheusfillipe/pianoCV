import { afterEach, beforeEach, describe, expect, test } from "vitest";
import type { Detection, Detector } from "./detector";
import { INPUT_SIZE } from "./detector";
import { applyHomography, findHomography, type Point } from "./homography";
import {
  cameraFocalFraction,
  canonicalQuad,
  keybedDepth,
  setCameraFocal,
  setKeybedDepth,
} from "./pose";
import type { Stillness } from "./stillness";
import { createTracker, type Measured } from "./tracker";

// projected from a real 118mm-deep, 52-white-key keybed at a plausible camera pose (pitch 12,
// yaw 10, focal 850, distance 950mm, 1280x720 frame), so it clears lockKeybed's shape, pose and
// black-key-stripe checks the way a real detection would
const BASE_QUAD: Point[] = [
  { x: 0.20881, y: 0.409305 },
  { x: 0.843085, y: 0.437863 },
  { x: 0.832954, y: 0.603702 },
  { x: 0.21614, y: 0.551411 },
];
const WIDTH = 1280;
const HEIGHT = 720;
const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

function shift(quad: Point[], dx: number, dy: number): Point[] {
  return quad.map((p) => ({ x: p.x + dx, y: p.y + dy }));
}

// paints exactly the pixels facing()'s band sampler reads: dark near the back edge (v < 0.4),
// light near the front edge (v > 0.6), so a synthetic quad reads as sitting on real black keys
function keybedGray(quad: Point[], size: number): Float32Array {
  const gray = new Float32Array(size * size).fill(0.5);
  const homography = findHomography(UNIT_SQUARE, canonicalQuad(quad));
  const steps = 300;
  for (let row = 0; row <= steps; row += 1) {
    const v = row / steps;
    const value = v < 0.4 ? 0.05 : v > 0.6 ? 0.95 : 0.5;
    for (let col = 0; col <= steps; col += 1) {
      const u = col / steps;
      const p = applyHomography(homography, u, v);
      const x = Math.round(p.x * size);
      const y = Math.round(p.y * size);
      if (x >= 0 && x < size && y >= 0 && y < size) {
        gray[y * size + x] = value;
      }
    }
  }
  return gray;
}

function keybedDetection(quad: Point[], still = true): Detection {
  return {
    quad,
    inputQuad: quad,
    motion: 0,
    still,
    mask: new Uint8Array(144 * 144),
    maskSize: 144,
    coverage: 1,
    confidence: 1,
    latencyMs: 5,
    gray: keybedGray(quad, INPUT_SIZE),
  };
}

function noKeybedDetection(): Detection {
  return {
    quad: null,
    inputQuad: null,
    motion: 0,
    still: true,
    mask: new Uint8Array(0),
    maskSize: 0,
    coverage: 0,
    confidence: 0,
    latencyMs: 1,
    gray: new Float32Array(0),
  };
}

interface FakeDetector extends Detector {
  calls: number;
}

function fakeDetector(script: (call: number) => Detection): FakeDetector {
  const fake: FakeDetector = {
    calls: 0,
    detect: async () => {
      const detection = script(fake.calls);
      fake.calls += 1;
      return detection;
    },
  };
  return fake;
}

interface FakeStillness extends Stillness {
  setChanged(value: boolean): void;
}

function fakeStillness(changed = true): FakeStillness {
  let current = changed;
  return {
    changed: () => current,
    forget: () => {},
    setChanged: (value) => {
      current = value;
    },
  };
}

function fakeVideo(): HTMLVideoElement {
  return {
    videoWidth: WIDTH,
    videoHeight: HEIGHT,
  } as unknown as HTMLVideoElement;
}

async function huntToHold(
  tracker: ReturnType<typeof createTracker>,
  frame: HTMLVideoElement,
  now: number,
): Promise<number> {
  for (let i = 0; i < 4; i += 1) {
    now += 400;
    await tracker.look(frame, now);
  }
  return now;
}

describe("tracker", () => {
  let restoreDepth: number;
  let restoreFocal: number;

  beforeEach(() => {
    restoreDepth = keybedDepth();
    restoreFocal = cameraFocalFraction();
  });

  afterEach(() => {
    setKeybedDepth(restoreDepth);
    setCameraFocal(restoreFocal);
  });

  test("holds after 4 agreeing reads and not after 3", async () => {
    const detector = fakeDetector(() => keybedDetection(BASE_QUAD));
    const tracker = createTracker(detector, { stillness: fakeStillness() });
    const frame = fakeVideo();
    let now = 1000;
    for (let i = 0; i < 3; i += 1) {
      now += 400;
      await tracker.look(frame, now);
    }
    expect(tracker.state().kind).toBe("hunting");

    now += 400;
    await tracker.look(frame, now);
    expect(tracker.state().kind).toBe("held");
    expect(tracker.reading()).toEqual({
      latencyMs: 5,
      coverage: 1,
      confidence: 1,
    });
  });

  test("does not hold when reads disagree", async () => {
    const detector = fakeDetector((call) =>
      keybedDetection(shift(BASE_QUAD, call * 0.03, 0), false),
    );
    const tracker = createTracker(detector, { stillness: fakeStillness() });
    const frame = fakeVideo();
    let now = 1000;
    for (let i = 0; i < 8; i += 1) {
      now += 400;
      await tracker.look(frame, now);
    }
    expect(tracker.state().kind).toBe("hunting");
  });

  test("never moves a held quad", async () => {
    const detector = fakeDetector((call) =>
      call < 4
        ? keybedDetection(BASE_QUAD)
        : keybedDetection(shift(BASE_QUAD, 0.03, 0)),
    );
    const tracker = createTracker(detector, { stillness: fakeStillness() });
    const frame = fakeVideo();
    const now = await huntToHold(tracker, frame, 1000);
    const held = tracker.state();
    if (held.kind !== "held") {
      throw new Error("expected held");
    }

    await tracker.look(frame, now + 3000);
    const after = tracker.state();
    if (after.kind !== "held") {
      throw new Error("expected still held");
    }
    expect(after.quad).toBe(held.quad);
  });

  test("loses after a move beyond 0.12", async () => {
    const detector = fakeDetector((call) =>
      call < 4
        ? keybedDetection(BASE_QUAD)
        : keybedDetection(shift(BASE_QUAD, 0, 0.2)),
    );
    let lost = 0;
    const tracker = createTracker(detector, {
      stillness: fakeStillness(),
      onLost: () => {
        lost += 1;
      },
    });
    const frame = fakeVideo();
    const now = await huntToHold(tracker, frame, 1000);
    expect(tracker.state().kind).toBe("held");

    await tracker.look(frame, now + 3000);
    expect(tracker.state()).toMatchObject({ kind: "lost" });
    expect(lost).toBe(1);
  });

  test("tolerates 19 misses while held and loses at 20", async () => {
    const detector = fakeDetector((call) =>
      call < 4 ? keybedDetection(BASE_QUAD) : noKeybedDetection(),
    );
    let lost = 0;
    const tracker = createTracker(detector, {
      stillness: fakeStillness(),
      onLost: () => {
        lost += 1;
      },
    });
    const frame = fakeVideo();
    let now = await huntToHold(tracker, frame, 1000);
    expect(tracker.state().kind).toBe("held");

    for (let i = 0; i < 19; i += 1) {
      now += 3000;
      await tracker.look(frame, now);
      expect(tracker.state().kind).toBe("held");
    }
    now += 3000;
    await tracker.look(frame, now);
    expect(tracker.state().kind).toBe("lost");
    expect(lost).toBe(1);
  });

  test("stops detecting while lost until release()", async () => {
    const detector = fakeDetector((call) =>
      call < 4 ? keybedDetection(BASE_QUAD) : noKeybedDetection(),
    );
    const tracker = createTracker(detector, { stillness: fakeStillness() });
    const frame = fakeVideo();
    let now = await huntToHold(tracker, frame, 1000);
    for (let i = 0; i < 20; i += 1) {
      now += 3000;
      await tracker.look(frame, now);
    }
    expect(tracker.state().kind).toBe("lost");

    const callsWhenLost = detector.calls;
    now += 3000;
    await tracker.look(frame, now);
    expect(detector.calls).toBe(callsWhenLost);
    expect(tracker.state().kind).toBe("lost");

    tracker.release();
    expect(tracker.state().kind).toBe("hunting");
    now += 400;
    await tracker.look(frame, now);
    expect(detector.calls).toBeGreaterThan(callsWhenLost);
  });

  test("hold() from hand calls onMeasured, and the detector path never does", async () => {
    const measured: Measured[] = [];
    const detector = fakeDetector(() => keybedDetection(BASE_QUAD));
    const tracker = createTracker(detector, {
      stillness: fakeStillness(),
      onMeasured: (m) => measured.push(m),
    });
    const frame = fakeVideo();
    // a look() call first, so the tracker has learned the frame size hold() measures against
    await tracker.look(frame, 1000);

    tracker.hold(BASE_QUAD);
    expect(measured).toHaveLength(1);
    expect(tracker.state()).toMatchObject({ kind: "held", byHand: true });

    tracker.release();
    await huntToHold(tracker, frame, 1000);
    expect(tracker.state()).toMatchObject({ kind: "held", byHand: false });
    expect(measured).toHaveLength(1);
  });

  test("does not run the model while held and still until 30 s pass", async () => {
    const detector = fakeDetector(() => keybedDetection(BASE_QUAD));
    const stillness = fakeStillness(true);
    const tracker = createTracker(detector, { stillness });
    const frame = fakeVideo();
    const now0 = await huntToHold(tracker, frame, 1000);
    expect(tracker.state().kind).toBe("held");
    const callsAtHold = detector.calls;

    stillness.setChanged(false);
    let now = now0 + 29000;
    await tracker.look(frame, now);
    expect(detector.calls).toBe(callsAtHold);

    now += 2000;
    await tracker.look(frame, now);
    expect(detector.calls).toBe(callsAtHold + 1);
    expect(tracker.state().kind).toBe("held");
  });
});
