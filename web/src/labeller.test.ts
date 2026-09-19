import { describe, expect, it } from "vitest";
import type { BoardRead } from "./board";
import type { Point } from "./homography";
import {
  createLabeller,
  type LabelSidecar,
  saveEveryMs,
  savesPerSession,
} from "./labeller";
import type { Stillness } from "./stillness";
import type { TrackerState } from "./tracker";

const QUAD: Point[] = [
  { x: 0.2, y: 0.4 },
  { x: 0.8, y: 0.42 },
  { x: 0.78, y: 0.6 },
  { x: 0.22, y: 0.58 },
];

function held(byHand: boolean): TrackerState {
  return { kind: "held", quad: QUAD, byHand };
}

function hunting(): TrackerState {
  return {
    kind: "hunting",
    progress: { agreed: 0, reason: "finding the keybed" },
  };
}

function lost(): TrackerState {
  return { kind: "lost", reason: "keybed moved out of place" };
}

function readAt(agreement: number): BoardRead {
  return {
    kind: "read",
    board: { lowest: 21, highest: 108, origin: 0, span: 52 },
    agreement,
    darkShare: 0.5,
    depth: 0.3,
  };
}

interface FakeStillness extends Stillness {
  setChanged(value: boolean): void;
}

function fakeStillness(changed = true): FakeStillness {
  let current = changed;
  return {
    changed: () => current,
    forget: () => undefined,
    setChanged: (value) => {
      current = value;
    },
  };
}

function fakeVideo(): HTMLVideoElement {
  return { videoWidth: 640, videoHeight: 480 } as unknown as HTMLVideoElement;
}

function fakeSessions(): () => string {
  let count = 0;
  return () => {
    count += 1;
    return `session-${count}`;
  };
}

function harness(stillness: Stillness = fakeStillness()): {
  labeller: ReturnType<typeof createLabeller>;
  saves: LabelSidecar[];
} {
  const saves: LabelSidecar[] = [];
  const labeller = createLabeller({
    save: (_frame, sidecar) => saves.push(sidecar),
    stillness,
    newSession: fakeSessions(),
  });
  return { labeller, saves };
}

describe("createLabeller", () => {
  it("saves a held board read at 0.9 agreement, not at 0.89", () => {
    const { labeller, saves } = harness();
    const frame = fakeVideo();
    const state = held(false);
    labeller.look(frame, state, readAt(0.89), true, 0);
    expect(saves).toHaveLength(0);
    labeller.look(frame, state, readAt(0.9), true, saveEveryMs);
    expect(saves).toHaveLength(1);
    expect(saves[0]).toMatchObject({
      kind: "auto",
      source: "board",
      agreement: 0.9,
    });
  });

  it("saves hand-placed corners with no board read at all", () => {
    const { labeller, saves } = harness();
    labeller.look(fakeVideo(), held(true), null, true, 0);
    expect(saves).toHaveLength(1);
    expect(saves[0].source).toBe("hand");
    expect(saves[0].agreement).toBeUndefined();
  });

  it("never saves while hunting or lost", () => {
    const { labeller, saves } = harness();
    labeller.look(fakeVideo(), hunting(), readAt(0.99), true, 0);
    labeller.look(fakeVideo(), lost(), readAt(0.99), true, saveEveryMs);
    expect(saves).toHaveLength(0);
  });

  it("never saves while the toggle is off", () => {
    const { labeller, saves } = harness();
    labeller.look(fakeVideo(), held(true), null, false, 0);
    expect(saves).toHaveLength(0);
  });

  it("waits at least saveEveryMs between saves of one hold", () => {
    const { labeller, saves } = harness();
    const frame = fakeVideo();
    const state = held(true);
    labeller.look(frame, state, null, true, 0);
    labeller.look(frame, state, null, true, saveEveryMs - 1);
    expect(saves).toHaveLength(1);
    labeller.look(frame, state, null, true, saveEveryMs);
    expect(saves).toHaveLength(2);
  });

  it("skips a save when the picture has not changed since the last one", () => {
    const stillness = fakeStillness(true);
    const { labeller, saves } = harness(stillness);
    const frame = fakeVideo();
    const state = held(true);
    labeller.look(frame, state, null, true, 0);
    expect(saves).toHaveLength(1);

    stillness.setChanged(false);
    labeller.look(frame, state, null, true, saveEveryMs);
    expect(saves).toHaveLength(1);

    stillness.setChanged(true);
    labeller.look(frame, state, null, true, saveEveryMs * 2);
    expect(saves).toHaveLength(2);
  });

  it("stops at 300 saves for the page session", () => {
    const { labeller, saves } = harness();
    const frame = fakeVideo();
    const state = held(true);
    for (let i = 0; i < savesPerSession + 5; i += 1) {
      labeller.look(frame, state, null, true, i * saveEveryMs);
    }
    expect(saves).toHaveLength(savesPerSession);
    expect(labeller.saved()).toBe(savesPerSession);
  });

  it("shares one session id across a hold and mints a new one for the next", () => {
    const { labeller, saves } = harness();
    const frame = fakeVideo();
    const firstHold = held(true);
    labeller.look(frame, firstHold, null, true, 0);
    labeller.look(frame, firstHold, null, true, saveEveryMs);
    expect(saves).toHaveLength(2);
    expect(saves[0].session).toBe(saves[1].session);

    const secondHold = held(true);
    labeller.look(frame, secondHold, null, true, saveEveryMs * 2);
    expect(saves).toHaveLength(3);
    expect(saves[2].session).not.toBe(saves[0].session);
  });
});
