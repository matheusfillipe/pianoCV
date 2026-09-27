import type { PerspectiveCamera } from "three";
import { Vector3 } from "three";
import type { Point } from "./homography";
export declare const KEY_TOP_Y = 0.61;
export declare const KEY_BOTTOM_Y = 0.0476;
export declare const BLACK_TOP_Y = 0.8902;
export declare const BACK_X = -1.627;
export declare const FRONT_X = 1.431;
export declare const SPAN_MIN_Z = -16.399;
export declare const SPAN_MAX_Z = 8.431;
export declare const MODEL_WHITE_KEYS = 52;
export declare const SPAN: number;
export declare const DEPTH: number;
export declare const WHITE_KEY_WIDTH_MM = 23.5;
export declare function mmToUnits(mm: number): number;
export interface KeybedCrop {
    readonly minZ: number;
    readonly maxZ: number;
}
export declare function keybedCrop(whiteKeys: number, startFactor: number): KeybedCrop;
export declare function cornersFor(minZ: number, maxZ: number): Vector3[];
export declare function centreFor(minZ: number, maxZ: number): Vector3;
export declare const KEYBED_CORNERS: Vector3[];
export declare const KEYBED_CENTRE: Vector3;
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
    readonly panelFaceX: number;
    readonly panelTopY: number;
    readonly panelInnerTopY: number;
}
export declare function caseLayout(crop: KeybedCrop, backDepthMm: number, cheekWidthMm: number): CaseLayout;
export interface PanelDetail {
    readonly size: readonly [number, number, number];
    readonly center: readonly [number, number, number];
    readonly lighter: number;
}
export declare function panelDetails(layout: CaseLayout, random: () => number): PanelDetail[];
export declare function projectCorners(camera: PerspectiveCamera, corners?: readonly Vector3[]): Point[];
export declare function visibleFraction(camera: PerspectiveCamera, corners?: readonly Vector3[]): number;
