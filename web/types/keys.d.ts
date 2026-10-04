export interface KeyUnits {
    from: number;
    to: number;
}
export declare function isBlack(pitch: number): boolean;
export declare function whiteIndex(pitch: number): number;
/** Where a key sits on any keyboard, in white-key widths from the same origin
 * `whiteIndex` counts from. The black keys carry the offsets a real instrument
 * has, where a black key straddles the join between two whites rather than
 * sitting over the middle of it. */
export declare function keyUnits(pitch: number): KeyUnits;
