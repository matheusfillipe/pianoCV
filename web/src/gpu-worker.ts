import * as ort from "onnxruntime-web/webgpu";

export type GpuRequest =
  | {
      readonly kind: "open";
      readonly id: number;
      readonly url: string;
      readonly wasm: string;
    }
  | {
      readonly kind: "keynet";
      readonly id: number;
      readonly url: string;
      readonly input: Float32Array;
      readonly dims: readonly number[];
    };

export type GpuReply =
  | {
      readonly kind: "opened";
      readonly id: number;
      /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
      readonly backend: string;
      /** Why the GPU could not run the model, when it could not. */
      readonly gpuFailure: string | null;
    }
  | { readonly kind: "failed"; readonly id: number; readonly reason: string }
  | {
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

const sessions = new Map<string, ort.InferenceSession>();

const post = (reply: GpuReply, transfer: Transferable[] = []): void => {
  self.postMessage(reply, { transfer });
};

async function open(id: number, url: string, wasm: string): Promise<void> {
  // the bundled build carries its own WebGPU loader, which an overridden script would replace
  ort.env.wasm.wasmPaths = { wasm };
  // a GPU adapter onnxruntime rejects fails the WebGPU start, and this worker's runtime then
  // falls back to wasm for that model
  let gpuFailure: string | null = null;
  for (const providers of [["webgpu"], ["wasm"]]) {
    try {
      const session = await ort.InferenceSession.create(url, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      sessions.set(url, session);
      post({ kind: "opened", id, backend: providers[0], gpuFailure });
      return;
    } catch (error) {
      if (providers[0] === "wasm") {
        post({
          kind: "failed",
          id,
          reason: `${gpuFailure} then ${String(error)}`,
        });
      } else {
        gpuFailure = String(error);
      }
    }
  }
}

async function runKeyNet(
  session: ort.InferenceSession,
  input: Float32Array,
  dims: readonly number[],
): Promise<KeyNetOutputs | null> {
  // the offset head trains to zero unless a recipe turns it on, so we skip reading it back
  const outputs = await session.run(
    { [session.inputNames[0]]: new ort.Tensor("float32", input, dims) },
    ["heatmaps", "presence"],
  );
  const heat = await outputs.heatmaps?.getData();
  const presence = await outputs.presence?.getData();
  if (!(heat instanceof Float32Array)) {
    return null;
  }
  return {
    heat,
    presence: presence instanceof Float32Array ? presence[0] : 0,
    offsets: null,
  };
}

// messages arrive one at a time and we await each, so the GPU never runs two inferences at once
let queue: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<GpuRequest>) => {
  const request = event.data;
  queue = queue.then(async () => {
    if (request.kind === "open") {
      await open(request.id, request.url, request.wasm);
      return;
    }
    const session = sessions.get(request.url);
    const outputs = session
      ? await runKeyNet(session, request.input, request.dims).catch(() => null)
      : null;
    post({ kind: "keynetresult", id: request.id, outputs });
  });
};
