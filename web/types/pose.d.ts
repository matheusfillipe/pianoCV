import { type Point } from "./homography";
export declare const WHITE_KEY_MM = 23.5;
export declare const KEYBED_DEPTH_MM = 118;
export declare const WHITE_KEY_COUNT: number;
export declare const DEPTH_UNITS: number;
/** A point of the keybed's own space, in white-key widths: u along the keys, v
 * across their depth from the player's edge, w standing off the plane. */
export interface Vector3 {
    u: number;
    v: number;
    w: number;
}
export interface PlanePose {
    focal: number;
    /** Millimetres per white-key width across the keys' depth, relative to along them: 1 when
     * the pose took the keybed's depth as given, the measured ratio when it read the depth off
     * the picture. */
    depthScale: number;
    rotation: number[][];
    translation: number[];
    worldWidthMm: number;
    residual: number;
    /** The sign that turns a height above the keybed into the rotation's third axis. */
    up: 1 | -1;
}
export declare function setKeybedDepth(units: number): void;
export declare function keybedDepth(): number;
export declare function setCameraFocal(fraction: number): void;
export declare function cameraFocalFraction(): number;
/** The keybed as the fit and the pose both see it, so a depth the user measures
 * moves them together. Corner 0 to 1 spans the keys and 1 to 2 the depth, with
 * the player at the near edge. */
export declare function worldCorners(): Point[];
export declare function canonicalQuad(quad: Point[]): Point[];
export declare function estimateFocal(imageCorners: Point[], width: number, height: number, world?: Point[]): number;
export declare function solvePose(imageCorners: Point[], width: number, height: number, world?: Point[]): PlanePose;
/** Where a point of the keybed's own space lands in the picture. u runs along
 * the keys and v across their depth, both in white-key widths, and w stands off
 * the plane, which is where anything drawn over the instrument lives. */
export declare function projectSpace(pose: PlanePose, u: number, v: number, w: number, width: number, height: number): Point;
export declare function projectPoint(pose: PlanePose, u: number, v: number, width: number, height: number): Point;
/** How far in front of the camera a point of the keybed's space sits, in
 * white-key widths. Anything at or behind zero has no picture to be drawn in. */
export declare function spaceDepth(pose: PlanePose, u: number, v: number, w: number): number;
/** Where the camera itself stands in the keybed's space, in white-key widths.
 * What is drawn above the keys is aimed at this. */
export declare function cameraPosition(pose: PlanePose): Vector3;
