import { type BoardRead, type Picture } from "./board";
import type { Board, KeybedSpace, PitchRange, Size } from "./keyspace";
export type Capture = (frame: CanvasImageSource, size: Size) => Picture | null;
export type BoardReader = {
    /** The board found so far, or null while the picture has not answered. */
    readonly board: () => Board | null;
    /** What the last try found, including why it was unsure, or null before
     * any try has run. */
    readonly last: () => BoardRead | null;
    /** Reads the keys again where a read is still owed. Cheap to call every
     * frame: it keeps its own clock and its own budget. A caller with a pinned
     * range skips the picture entirely and takes it as the board. */
    readonly look: (space: KeybedSpace, frame: CanvasImageSource, size: Size, now: number, pinned?: PitchRange | null) => void;
    /** The corners changed, so the last answer was about a different keybed.
     * The keys are read again once they stop moving. */
    readonly moved: (now: number) => void;
    /** A note the player sounded. Colour says how many keys there are and which
     * note the board starts on inside an octave, never which octave, so notes
     * played narrow which octave the next read may answer. */
    readonly played: (pitch: number) => void;
};
export declare function createBoardReader(capture?: Capture): BoardReader;
