import { type Point } from "./homography";
export declare function estimateFocalFraction(quad: Point[], width: number, height: number): number;
export declare function keybedDepthFromQuad(quad: Point[], focal: number, cx: number, cy: number): number;
export declare const FOCAL_SCAN: number[];
export declare let lastDecline: string;
export interface RectFit {
    quad: Point[];
    rectangle: Point[];
    cost: number;
    focal: number;
}
export declare function boundaryResidual(points: Point[], quad: Point[]): number;
export declare function boundaryPoints(probability: Float32Array, size: number, width: number, height: number, near: Point[] | null): Point[];
export declare function snapToGradient(gray: Float32Array, width: number, height: number, points: Point[], quad: Point[]): Point[];
export declare function principalBox(points: Point[]): Point[];
export interface Frame {
    gray: Float32Array;
    width: number;
    height: number;
}
export declare function endMovesOnlyInward(quad: Point[], end: number, moves: readonly {
    corner: number;
    point: Point;
}[]): boolean;
export declare function refineEnds(frame: Frame, quad: Point[]): Point[];
export declare function fitRectangle(points: Point[], coarseQuad: Point[], width: number, height: number, focal?: number, frame?: Frame): RectFit | null;
