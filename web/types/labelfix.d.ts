import { type KeyNetFit, type KeyNetPeaks } from "./keycore";
import type { Size } from "./keyspace";
/** A point in frame pixels, or null where it is hidden by a hand, the case, or off frame. */
export type LabelPoint = [number, number] | null;
export declare const POINT_KINDS: readonly ["corners", "gaps", "blackLow", "blackHigh", "blackTopLow", "blackTopHigh", "backGaps", "blackBackLow", "blackBackHigh"];
export type PointKind = (typeof POINT_KINDS)[number];
export type FixedLabels = Record<PointKind, LabelPoint[]> & {
    width: number;
    height: number;
    whiteKeys: number;
    phase: string;
};
type Locator = Pick<KeyNetFit, "homography" | "whiteKeys" | "phase" | "lift">;
type FrontTops = Pick<KeyNetPeaks, "blackTopLow" | "blackTopHigh">;
type BackPeaks = Pick<KeyNetPeaks, "blackBackLow" | "blackBackHigh">;
type RearKind = "backGaps" | "blackBackLow" | "blackBackHigh";
export type RearLabels = Pick<FixedLabels, RearKind>;
export declare function emptyLabels(size: Size): FixedLabels;
export declare function serialise(labels: FixedLabels): string;
/** Files saved before the rear kinds existed parse with those kinds all hidden. */
export declare function parseLabels(text: string): FixedLabels;
export declare function lacksRear(text: string): boolean;
/** The rear labels the fit implies: back gaps on the template through the homography, and for each
 * black key the back-top peak nearest its lifted template point within one white key, else that
 * lifted point itself when the fit has a lift. */
export declare function prefillRear(fit: Locator, peaks: BackPeaks, size: Size): RearLabels;
/** The labels the app's own fit implies: the template through the fit's homography, and for each
 * black key the nearest top peak within one white key of its bottom. */
export declare function prefill(fit: Locator, tops: FrontTops & BackPeaks, size: Size): FixedLabels;
/** Moves every gap and black point `keys` template indices along, through a least-squares
 * homography from the template onto the points that are visible now; 0 refits them in place.
 * Hidden points stay null unless `fillHidden`, which gives them a projected position instead. */
export declare function reproject(labels: FixedLabels, keys: number, fillHidden?: boolean): FixedLabels;
export {};
