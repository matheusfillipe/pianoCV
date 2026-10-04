import { CanvasTexture } from "three";
export type BackdropKind = "clutter" | "noise" | "gradient";
export declare function makeFloorTexture(random: () => number): CanvasTexture;
export declare function makeBackdrop(kind: BackdropKind, random: () => number): CanvasTexture;
