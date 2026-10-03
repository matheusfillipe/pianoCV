import type { RuntimeAssets } from "./assets";
import type { KeyNetOutputs } from "./gpu-worker";
export declare const KEYNET_URL = "/keynet.onnx";
export type KeyNetRunner = {
    /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
    readonly backend: string;
    readonly gpuFailure: string | null;
    /** The model's raw outputs for one NCHW input, or null when it gave none. */
    readonly run: (input: Float32Array, width: number, height: number) => Promise<KeyNetOutputs | null>;
};
export declare function createKeyNet(assets: RuntimeAssets, url?: string): Promise<KeyNetRunner>;
