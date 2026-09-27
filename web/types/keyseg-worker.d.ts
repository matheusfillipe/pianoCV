import { type Crop, type KeyRegion } from "./keyseg";
export type SegmentRequest = {
    readonly kind: "init";
    readonly url: string;
    readonly wasm: string;
} | {
    readonly kind: "segment";
    readonly id: number;
    readonly frame: ImageBitmap;
    readonly crop: Crop;
};
export type SegmentReply = {
    readonly kind: "ready";
    readonly backend: string;
    /** Why the GPU could not run the model, when it could not. */
    readonly gpuFailure: string | null;
} | {
    readonly kind: "failed";
    readonly reason: string;
} | {
    readonly kind: "segmented";
    readonly id: number;
    /** The outline's corners in frame pixels, or null when too few key pixels were found. */
    readonly corners: {
        x: number;
        y: number;
    }[] | null;
    /** Every key region, in frame pixels. */
    readonly regions: KeyRegion[];
};
