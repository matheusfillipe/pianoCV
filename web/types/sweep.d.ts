import type { PitchRange } from "./keyspace";
export interface Range {
    readonly min: number;
    readonly max: number;
}
export declare function between(random: () => number, range: Range): number;
export declare const SWEEP_ELEVATION_DEG: Range;
export declare const SWEEP_AZIMUTH_DEG: Range;
export declare const SWEEP_DISTANCE_FACTOR: Range;
export declare const SWEEP_ROLL_DEG: Range;
export declare const SWEEP_FOV_DEG: Range;
export declare const SWEEP_OFFSET_X_FACTOR: Range;
export declare const SWEEP_OFFSET_Y_FACTOR: Range;
export interface SweepPose {
    readonly elevationDeg: number;
    readonly azimuthDeg: number;
    readonly distanceFactor: number;
    readonly rollDeg: number;
    readonly fovDeg: number;
    readonly offsetXFactor: number;
    readonly offsetYFactor: number;
}
export declare function samplePose(random: () => number): SweepPose;
export declare function noteName(pitch: number): string;
export interface BoardSize {
    readonly keys: number;
    readonly whiteKeys: number;
    readonly lowestNote: string;
    readonly range: PitchRange;
}
export declare const BOARD_SIZES: readonly BoardSize[];
export declare function pickBoardSize(random: () => number): BoardSize;
export declare const CASE_PRESENCE_PROBABILITY = 0.9;
export declare const CASE_BACK_DEPTH_MM: Range;
export declare const CASE_TOP_STANDOFF_MM: Range;
export declare const CASE_CHEEK_WIDTH_MM: Range;
export type CaseColorFamily = "black" | "darkGrey" | "silver" | "white" | "wood";
export interface CaseColorSpec {
    readonly family: CaseColorFamily;
    readonly weight: number;
    readonly hue: Range;
    readonly saturation: Range;
    readonly lightness: Range;
    readonly roughness: Range;
    readonly metalness: Range;
}
export declare function pickCaseColorFamily(random: () => number): CaseColorSpec;
