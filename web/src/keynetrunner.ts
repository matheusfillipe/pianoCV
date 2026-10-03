import type { RuntimeAssets } from "./assets";
import { openModel } from "./gpu";
import type { KeyNetOutputs } from "./gpu-worker";
import { loadKeycore } from "./keycore";

export const KEYNET_URL = "/keynet.onnx";

export type KeyNetRunner = {
  /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
  readonly backend: string;
  readonly gpuFailure: string | null;
  /** The model's raw outputs for one NCHW input, or null when it gave none. */
  readonly run: (
    input: Float32Array,
    width: number,
    height: number,
  ) => Promise<KeyNetOutputs | null>;
};

export async function createKeyNet(
  assets: RuntimeAssets,
  url: string = KEYNET_URL,
): Promise<KeyNetRunner> {
  const [model] = await Promise.all([openModel(assets, url), loadKeycore()]);
  return {
    backend: model.backend,
    gpuFailure: model.gpuFailure,
    run: async (input, width, height) => {
      const reply = await model.worker.ask(
        {
          kind: "keynet",
          url: model.url,
          input,
          dims: [1, 3, height, width],
        },
        [input.buffer],
      );
      return reply.kind === "keynetresult" ? reply.outputs : null;
    },
  };
}
