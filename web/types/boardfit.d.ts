import { type Point } from "./homography";
import type { Board } from "./keypolygons";
export declare function fitBoard(pixels: ImageData, quad: readonly Point[]): Board | null;
