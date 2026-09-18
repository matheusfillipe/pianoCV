import { type Point } from "./homography";
export interface KeyPolygon {
    pitch: number;
    black: boolean;
    points: Point[];
}
export interface BlackKeyGeometry {
    pitch: number;
    u0: number;
    u1: number;
    depth: number;
    confidence: number;
}
export interface Board {
    lowest: number;
    highest: number;
    origin: number;
    span: number;
    blackDepth: number;
    blackKeys?: readonly BlackKeyGeometry[];
}
export declare function keyPolygons(quad: readonly Point[], board: Board): KeyPolygon[];
export declare function drawKeyMasks(ctx: CanvasRenderingContext2D, quad: readonly Point[], board: Board, width: number, height: number): void;
