export const stillDirs = ["synth", "synth-case", "synth-keys"] as const;
export const motionDirs = ["synth-motion", "synth-test"] as const;

export type StillDir = (typeof stillDirs)[number];
export type MotionDir = (typeof motionDirs)[number];

export function stillDirOf(out: string): StillDir | null {
  return stillDirs.find((dir) => dir === out) ?? null;
}

export function motionDirOf(out: string): MotionDir | null {
  return motionDirs.find((dir) => dir === out) ?? null;
}
