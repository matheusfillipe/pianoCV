export type GpuRequest = {
    readonly kind: "open";
    readonly id: number;
    readonly url: string;
    readonly wasm: string;
} | {
    readonly kind: "keynet";
    readonly id: number;
    readonly url: string;
    readonly input: Float32Array;
    readonly dims: readonly number[];
};
export type GpuReply = {
    readonly kind: "opened";
    readonly id: number;
    /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
    readonly backend: string;
    /** Why the GPU could not run the model, when it could not. */
    readonly gpuFailure: string | null;
} | {
    readonly kind: "failed";
    readonly id: number;
    readonly reason: string;
} | {
    readonly kind: "keynetresult";
    readonly id: number;
    /** The model's raw outputs, or null when it gave no heatmaps. */
    readonly outputs: KeyNetOutputs | null;
};
export type KeyNetOutputs = {
    readonly heat: Float32Array;
    readonly presence: number;
    readonly offsets: Float32Array | null;
};
