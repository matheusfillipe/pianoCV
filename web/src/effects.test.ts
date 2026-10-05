import { describe, expect, it } from "vitest";
import {
  createEffectsState,
  glowLevel,
  lowestPitchFor,
  noteOff,
  noteOn,
  prune,
} from "./effects";

describe("lowestPitchFor", () => {
  it("maps every standard board to its lowest pitch", () => {
    expect(lowestPitchFor("C", 48)).toBe(36);
    expect(lowestPitchFor("C", 60)).toBe(36);
    expect(lowestPitchFor("E", 75)).toBe(28);
    expect(lowestPitchFor("A", 87)).toBe(21);
  });

  it("centres a board of any other size where a full piano is centred", () => {
    // A 25-key board from a C sits in the middle of the piano, C3 to C5.
    expect(lowestPitchFor("C", 24)).toBe(48);
  });

  it("numbers nothing for a letter that is not a key", () => {
    expect(lowestPitchFor("H", 48)).toBeNull();
  });
});

describe("glow state", () => {
  it("jumps to a level from velocity on note on", () => {
    const state = noteOn(createEffectsState(), 64, 127, 0);
    expect(glowLevel(state, 64, 0)).toBeCloseTo(1);
    const half = noteOn(createEffectsState(), 64, 64, 0);
    expect(glowLevel(half, 64, 0)).toBeCloseTo(64 / 127);
  });

  it("holds its level while the note stays down", () => {
    const state = noteOn(createEffectsState(), 64, 100, 0);
    expect(glowLevel(state, 64, 5000)).toBeCloseTo(100 / 127);
  });

  it("fades out linearly over 400ms after release", () => {
    const held = noteOn(createEffectsState(), 64, 127, 0);
    const released = noteOff(held, 64, 1000);
    expect(glowLevel(released, 64, 1000)).toBeCloseTo(1);
    expect(glowLevel(released, 64, 1200)).toBeCloseTo(0.5);
    expect(glowLevel(released, 64, 1400)).toBe(0);
    expect(glowLevel(released, 64, 2000)).toBe(0);
  });

  it("treats velocity 0 as a release", () => {
    const held = noteOn(createEffectsState(), 64, 127, 0);
    const released = noteOn(held, 64, 0, 1000);
    expect(glowLevel(released, 64, 1000)).toBeCloseTo(1);
    expect(glowLevel(released, 64, 1400)).toBe(0);
  });

  it("leaves an unplayed pitch dark", () => {
    expect(glowLevel(createEffectsState(), 64, 0)).toBe(0);
  });

  it("prunes a fully faded note but keeps one still fading", () => {
    const held = noteOn(createEffectsState(), 64, 127, 0);
    const released = noteOff(held, 64, 0);
    const pruned = prune(released, 400);
    expect(pruned.notes.has(64)).toBe(false);

    const stillFading = prune(released, 200);
    expect(stillFading.notes.has(64)).toBe(true);
  });
});
