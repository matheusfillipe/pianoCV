import { type Point } from "./homography";
import type { KeyMatcher } from "./keymatch";
import type { KeySegmenter } from "./keyseg";
import type { Size } from "./keyspace";
import { type Bar } from "./keyspace";
declare global {
    interface Window {
        pianocvKeyStrip?: {
            width: number;
            height: number;
            data: Uint8ClampedArray;
        };
        pianocvKeyRead?: KeyRead;
        pianocvKeyEvidence?: KeyEvidence | null;
        pianocvReadFits?: {
            confidence: number;
            fit: number;
            chosen: boolean;
        }[];
    }
}
export declare const STRIP_WIDTH = 1200;
export declare const STRIP_HEIGHT = 150;
/** Heights a black key's top may stand above the white keys, either way round since which side
 * of the plane the camera sits on depends on how the corners wind; the fit picks the one that
 * lets white and black evidence agree. */
export declare const RAISE_CANDIDATES_MM: number[];
/** The sizes keyboards are built in, told apart by the letter of their first white key, with the
 * MIDI note that first key plays. */
export declare const STANDARD_BOARDS: readonly [{
    readonly phase: "C";
    readonly whiteKeys: 29;
    readonly lowestPitch: 36;
}, {
    readonly phase: "C";
    readonly whiteKeys: 36;
    readonly lowestPitch: 36;
}, {
    readonly phase: "E";
    readonly whiteKeys: 45;
    readonly lowestPitch: 28;
}, {
    readonly phase: "A";
    readonly whiteKeys: 52;
    readonly lowestPitch: 21;
}];
export type SourceImage = {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
};
export type Strip = {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
};
/** Where the keys begin, as a depth fraction at each end of the board, so the trimmed edge can
 * follow a case boundary that is deeper at one end. */
export type FarEdge = {
    readonly left: number;
    readonly right: number;
};
export type DetectedKey = {
    readonly bar: Bar;
    readonly black: boolean;
    /** Semitones above the board's first white key, which is what maps a MIDI note onto it. */
    readonly semitone: number;
};
export type KeyRead = {
    readonly kind: "read";
    readonly whiteKeys: number;
    readonly totalKeys: number;
    /** The white key the strip's left edge starts on, C to B. */
    readonly phase: string;
    readonly confidence: number;
    /** How many white keys wide the strip was, which is the rectangle the pose is solved
     * against; wider than the board when the mask ran past its ends. */
    readonly stripKeys: number;
    /** How far the black keys' tops stand off the keybed, as the fit measured it, in the
     * pose's own sign. */
    readonly blackRaiseMm: number;
    /** How far, in strip pixels along the keys, the black keys the strip shows sit from where
     * the pose puts them, as a line over the strip; the pose's lens and tilt are estimated from
     * one rectangle, so we pin the drawn black keys to what the evidence shows. */
    readonly blackShift: Line;
    /** The corners, in strip pixels of the quad the read was asked about, of the outline the
     * keys were actually read on, once its ends were squared to the keys. */
    readonly outline: readonly Point[];
    readonly keyAt: (index: number) => DetectedKey | null;
} | {
    readonly kind: "unsure";
    readonly reason: string;
    readonly confidence: number;
};
export declare function sampleBilinear(source: SourceImage, x: number, y: number): [number, number, number] | null;
/** Warps the held quad flat: the far edge, where the black keys end, lands at the top, and the
 * player's edge lands at the bottom, so the rest of this module can work in undistorted pixels. */
export declare function rectifyStrip(source: SourceImage, quad: readonly Point[], width?: number, height?: number): Strip | null;
/** Where a strip point at (x, y) appears in the same strip once raised `raiseMm` off the keybed:
 * the flat strip shows anything standing above the keys displaced by parallax. */
export type Lift = (x: number, y: number, raiseMm: number) => number;
/** A lift for a board of `whiteKeys`, whose width sets the aspect the pose is solved against. */
export type LiftFor = (whiteKeys: number) => Lift;
export type LinePoint = {
    readonly x: number;
    readonly y: number;
};
export type Line = {
    readonly meanX: number;
    readonly meanY: number;
    readonly slope: number;
};
export declare function fitLine(points: readonly LinePoint[]): Line;
export declare function lineAt(line: Line, x: number): number;
export declare function lineResidual(line: Line, point: LinePoint): number;
/** A strip's own corners, in strip pixels: the outline of a read made on the quad as given. */
export declare const PLAIN_OUTLINE: readonly Point[];
/** The quad whose strip corners are `outline` in the strip of `quad`. */
export declare function outlineQuad(quad: readonly Point[], outline: readonly Point[]): Point[];
export declare function liftFor(quad: readonly Point[], frame: Size): LiftFor;
/** The visible faces of every key in the current frame, as frame fractions: white keys flat on
 * the keybed, and each black key as its raised top plus the front face that drops to the white
 * keys, since no outline flat on the keybed can cover a key standing above it. */
export declare function projectKeyFaces(read: KeyRead, quad: readonly Point[], frame: Size): DetectedKey[];
/** Where the keys begin, measured per column band and fitted as a line, so a case boundary that
 * sits deeper at one end of the board is described by a tilt rather than collapsed to one depth
 * that can only ever be right in the middle. */
export declare function measureFarEdge(strip: Strip): FarEdge | null;
/** Moves the far edge (corners 0 and 1) toward the near edge (corners 2 and 3) by a fraction of
 * the depth, in the plane the quad is already drawn in. The near edge never moves: only the far
 * edge was ever measured onto the case instead of the keys. */
export declare function trimFarEdge(quad: readonly Point[], edge: FarEdge | null): Point[];
export type Run = {
    readonly start: number;
    readonly end: number;
    readonly center: number;
};
/** Where a strip shows its keys, in strip pixels: the gaps between white keys and the black keys'
 * extents. The brightness rules produce it, and a learned key matcher can stand in for them. */
export type KeyEvidence = {
    readonly dips: readonly number[];
    readonly runs: readonly Run[];
};
/** Finds the keys in a rectified strip: the white-key separator lines near the player's edge and
 * the black-key runs near the far edge, walked and refined rather than fit to one assumed grid,
 * so a leftover lens warp does not have to be modelled to be tolerated. */
export declare function detectKeys(strip: Strip, lift?: LiftFor | null, evidence?: KeyEvidence | null): KeyRead;
export declare function captureSource(frame: CanvasImageSource, size: Size): SourceImage | null;
/** A board read this unsure has been wrong about the key count on our recordings, and keys
 * drawn off by one would light the wrong note, so nothing draws a read below it. */
export declare const TRUSTED_READ = 0.5;
/** The median of the latest far-edge measurements, once enough of them agree at both ends of the
 * board; null while they are too few or still disagree. */
export declare function trustedEdge(recent: readonly FarEdge[]): FarEdge | null;
export type KeyReader = {
    readonly last: () => KeyRead | null;
    /** The far edge this reader currently trusts, or null before any try has found evidence. */
    readonly fraction: () => FarEdge | null;
    readonly look: (frame: CanvasImageSource, quad: readonly Point[], size: Size, now: number) => void;
    readonly moved: (now: number) => void;
};
type BoardRead = Extract<KeyRead, {
    kind: "read";
}>;
/** The grey level that best splits the values into a dark and a bright class. */
export declare function otsu(values: readonly number[]): number;
/** How well a read's keys sit on the picture it was read from: the share of points on its black
 * keys that are dark and on the front of its white keys that are bright, 1 when every point
 * agrees. Two reads of one board can agree on its keys and still place them half a key apart. */
export declare function readFit(read: BoardRead, quad: readonly Point[], source: SourceImage): number;
/** The read to trust across several looks at the same held keyboard: the board most looks
 * agree on, then the look among them that scores best, the most confident unless told how else
 * to score. A still camera shows the same keyboard every time, so a look that disagrees with the
 * rest caught a hand or a blur, and one lucky confident look should not outvote the others. */
export declare function boardConsensus(reads: readonly BoardRead[], score?: (read: BoardRead) => number): BoardRead | null;
/** Reads the keys with the learned matcher when one is given, and with the brightness rules
 * whenever the matcher is missing or sees too little. A key segmenter, when given, first turns
 * the held quad into the outline the key pixels show, so the template is laid on the keys the
 * camera sees rather than on the mask's guess. */
export declare function createKeyReader(matcher?: KeyMatcher | null, segmenter?: KeySegmenter | null): KeyReader;
export {};
