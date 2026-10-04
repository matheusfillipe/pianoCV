import type { Point } from "./homography";
export type Corners = [Point, Point, Point, Point];
/** Rolls the quad so edge 0 to 1 is its longer side, which runs along the white keys. The
 * remaining half turn is the dragger's choice: handle 1 to 2 runs along the black keys. */
export declare function canonicalQuad(quad: Point[]): Point[];
export interface Box {
    x: number;
    y: number;
    w: number;
    h: number;
}
export interface Calibration {
    getCorners(): Corners;
    setCorners(corners: readonly Point[]): void;
    draw(ctx: CanvasRenderingContext2D, w: number, h: number): void;
}
export declare function createCalibration(canvas: HTMLCanvasElement, isActive: () => boolean, getBox: () => Box): Calibration;
