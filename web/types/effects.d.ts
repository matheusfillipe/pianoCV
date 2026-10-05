import type { Point } from "./homography";
declare global {
    interface Window {
        pianocvPlay?: (pitch: number, velocity: number) => void;
    }
}
/** The MIDI pitch of the lowest key of a board with this many white keys starting on this
 * letter. We take the octave that centres the board where a full piano is centred, which is
 * where every standard size sits, since one C looks exactly like the next. */
export declare function lowestPitchFor(whiteKeys: number, phase: string): number | null;
type NoteState = {
    readonly velocity: number;
    readonly releasedAt: number | null;
};
export type EffectsState = {
    readonly notes: ReadonlyMap<number, NoteState>;
};
export declare function createEffectsState(): EffectsState;
export declare function noteOn(state: EffectsState, pitch: number, velocity: number, now: number): EffectsState;
export declare function noteOff(state: EffectsState, pitch: number, now: number): EffectsState;
/** How lit a pitch is right now: full from its velocity while held, fading linearly to
 * nothing over FADE_MS once released. */
export declare function glowLevel(state: EffectsState, pitch: number, now: number): number;
/** Drops notes that have fully faded, so the map does not grow for as long as the page runs. */
export declare function prune(state: EffectsState, now: number): EffectsState;
export type Effects = {
    readonly draw: (ctx: CanvasRenderingContext2D, faces: readonly {
        readonly black: boolean;
        readonly semitone: number;
        readonly bar: readonly Point[];
    }[], lowestPitch: number | null, w: number, h: number, now: number) => void;
};
export declare function createEffects(): Effects;
export {};
