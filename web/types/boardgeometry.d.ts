import { type Point } from "./homography";
import type { Board } from "./keypolygons";
export declare function refineBoardEdges(image: ImageData, quad: readonly Point[], board: Board): Point[];
export declare function calibrateBoard(image: ImageData, proposal: readonly Point[]): {
    board: Board;
    quad: Point[];
} | null;
