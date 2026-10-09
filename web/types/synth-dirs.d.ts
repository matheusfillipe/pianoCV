export declare const stillDirs: readonly ["synth", "synth-case", "synth-keys"];
export declare const motionDirs: readonly ["synth-motion", "synth-test"];
export type StillDir = (typeof stillDirs)[number];
export type MotionDir = (typeof motionDirs)[number];
export declare function stillDirOf(out: string): StillDir | null;
export declare function motionDirOf(out: string): MotionDir | null;
