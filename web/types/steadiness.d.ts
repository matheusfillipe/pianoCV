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
    /** How well each drawn black key overlaps the black key the segmenter found under it, over
     * the keys the segmenter found on their own. */
    readonly blackIoUMedian: number | null;
    readonly blackIoUP10: number | null;
    /** The same overlap for the template keys before they were snapped. */
    readonly templateIoUMedian: number | null;
    readonly templateIoUP10: number | null;
    /** The share of black keys the segmenter found on their own, unmerged. */
    readonly blackSingleShare: number | null;
    /** For the drawn keys and the template they were snapped from, the share of each black key
     * that is dark in the video and of the white keys' visible surface that is bright, split at
     * the keybed's own Otsu threshold. */
    readonly fit: Readonly<Record<string, FitSummary>>;
};
type FitSummary = {
    readonly blackDarkMedian: number | null;
    readonly blackDarkP10: number | null;
    readonly whiteBright: number | null;
};
export declare function measureSteadiness(durationMs?: number, settleMs?: number): Promise<Steadiness>;
export {};
