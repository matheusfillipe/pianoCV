/** How still and how well placed the drawn keys are on a live page, for the lab: the camera of
 * a recording does not move, so any motion of a drawn key is the pipeline's own. */
export type Steadiness = {
    readonly frames: number;
    readonly keys: number;
    /** How far a drawn key's outline moves from one frame to the next, in video pixels: each point
     * of it against the nearest point of the same key's outline the frame before. */
    readonly movePxMedian: number | null;
    readonly movePxP95: number | null;
    readonly movePxP999: number | null;
    /** The share of frames in which some key corner moved more than a pixel. */
    readonly jumpyFrames: number | null;
    /** How far each key's centre wanders around its own mean, in video pixels. */
    readonly swingPxMedian: number | null;
    readonly swingPxP95: number | null;
    /** The share of each black key that is dark in the video and of the white keys' visible surface
     * that is bright, split at the keybed's own Otsu threshold. */
    readonly fit: FitSummary;
};
type FitSummary = {
    readonly blackDarkMedian: number | null;
    readonly blackDarkP10: number | null;
    readonly whiteBright: number | null;
};
export declare function measureSteadiness(durationMs?: number, settleMs?: number): Promise<Steadiness>;
export {};
