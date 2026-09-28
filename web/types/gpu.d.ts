import type { RuntimeAssets } from "./assets";
import type { GpuReply, GpuRequest } from "./gpu-worker";
type Request = GpuRequest extends infer R ? R extends GpuRequest ? Omit<R, "id"> : never : never;
export type GpuWorker = {
    readonly wasm: string;
    readonly ask: (request: Request, transfer?: Transferable[]) => Promise<GpuReply>;
};
export type RemoteModel = {
    /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
    readonly backend: string;
    readonly gpuFailure: string | null;
    readonly url: string;
    readonly worker: GpuWorker;
    /** The model's first output for one float32 input, or null when it gave none. */
    readonly run: (input: Float32Array, dims: readonly number[]) => Promise<Float32Array | null>;
};
export declare function openModel(assets: RuntimeAssets, url: string): Promise<RemoteModel>;
export {};
