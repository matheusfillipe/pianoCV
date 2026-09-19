import { type Board, type KeybedSpace, type PitchRange } from "./keyspace";
/** A frame to read; it can be smaller than the display frame since colour
 * needs far less detail than the picture on screen. */
export type Picture = {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
    /** How much of the frame's size this picture is. */
    readonly scale: number;
};
export type BoardRead = {
    readonly kind: "read";
    readonly board: Board;
    /** Share of the readable samples whose colour matched the board found. */
    readonly agreement: number;
    /** Share of the samples that came back dark, and how deep into the
     * keybed they were taken. */
    readonly darkShare: number;
    readonly depth: number;
} | {
    readonly kind: "unsure";
    readonly reason: string;
};
/** Which keyboard is in front of the camera, read off its black keys: their
 * spacing gives the key count and phase, and every depth into the keybed is
 * tried since a low camera compresses the far edge unpredictably. Colour
 * alone can't tell one octave from another, so notes already played narrow
 * it, and otherwise we centre on a full piano. */
export declare function readBoard(space: KeybedSpace, picture: Picture, played?: PitchRange | null): BoardRead;
