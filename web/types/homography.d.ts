export interface Point {
    x: number;
    y: number;
}
export type Homography = number[];
export declare function solve(matrix: number[][], rhs: number[]): number[];
export declare function findHomography(src: readonly Point[], dst: readonly Point[]): Homography;
export declare function applyHomography(h: Homography, x: number, y: number): Point;
