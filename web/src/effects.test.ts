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
    expect(lowestPitchFor(29, "C")).toBe(36);
    expect(lowestPitchFor(36, "C")).toBe(36);
    expect(lowestPitchFor(45, "E")).toBe(28);
    expect(lowestPitchFor(52, "A")).toBe(21);
  });

  it("returns null for a board that matches no standard size", () => {
    expect(lowestPitchFor(40, "C")).toBeNull();
    expect(lowestPitchFor(29, "D")).toBeNull();
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
