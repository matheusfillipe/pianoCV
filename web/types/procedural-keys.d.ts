import { Vector3 } from "three";
import { type Range } from "./sweep";
export interface KeyboardBoard {
    readonly lowestPitch: number;
    readonly keys: number;
}
export interface KeyVariation {
    readonly blackWidthFrac: number;
    readonly blackOffsetJitter: readonly [number, number, number, number, number];
    readonly blackHeightMm: number;
    readonly blackLengthFrac: number;
    readonly whiteGapMm: number;
    readonly bevelMm: number;
}
export declare const BLACK_WIDTH_FRAC: Range;
export declare const BLACK_OFFSET_JITTER: Range;
export declare const BLACK_HEIGHT_MM: Range;
export declare const BLACK_LENGTH_FRAC: Range;
export declare const WHITE_GAP_MM: Range;
export declare const BEVEL_MM: Range;
export declare function sampleKeyVariation(random: () => number): KeyVariation;
export interface KeyMeshBox {
    readonly size: readonly [number, number, number];
    readonly center: readonly [number, number, number];
}
export interface KeyGeometry {
    readonly pitch: number;
    readonly black: boolean;
    readonly body: KeyMeshBox;
    readonly cap: KeyMeshBox;
    readonly topCorners: readonly Vector3[];
    readonly frontCorners: readonly Vector3[] | null;
}
export interface Keyboard {
    readonly board: KeyboardBoard;
    readonly whiteKeys: number;
    readonly variation: KeyVariation;
    readonly keys: readonly KeyGeometry[];
    readonly minZ: number;
    readonly maxZ: number;
}
export declare function buildKeyboard(board: KeyboardBoard, variation: KeyVariation): Keyboard;
