import type { RuntimeAssets } from "./assets";
import type { GpuReply, GpuRequest } from "./gpu-worker";

type Request = GpuRequest extends infer R
  ? R extends GpuRequest
    ? Omit<R, "id">
    : never
  : never;

export type GpuWorker = {
  readonly wasm: string;
  readonly ask: (
    request: Request,
    transfer?: Transferable[],
  ) => Promise<GpuReply>;
};

export type RemoteModel = {
  /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
  readonly backend: string;
  readonly gpuFailure: string | null;
  readonly url: string;
  readonly worker: GpuWorker;
  /** The model's first output for one float32 input, or null when it gave none. */
  readonly run: (
    input: Float32Array,
    dims: readonly number[],
  ) => Promise<Float32Array | null>;
};

// every model shares one worker, so they share one WebGPU device and one runtime
let shared: GpuWorker | null = null;

function gpuWorker(assets: RuntimeAssets): GpuWorker {
  if (shared) {
    return shared;
  }
  const worker = new Worker(new URL("./gpu-worker.ts", import.meta.url), {
    type: "module",
  });
  const pending = new Map<number, (reply: GpuReply) => void>();
  let nextId = 0;
  worker.onmessage = (event: MessageEvent<GpuReply>) => {
    pending.get(event.data.id)?.(event.data);
    pending.delete(event.data.id);
  };
  shared = {
    wasm: new URL(assets.ortGpuWasm, location.href).href,
    ask: (request, transfer = []) => {
      nextId += 1;
      const id = nextId;
      return new Promise((resolve) => {
        pending.set(id, resolve);
        worker.postMessage({ ...request, id }, transfer);
      });
    },
  };
  return shared;
}

export async function openModel(
  assets: RuntimeAssets,
  url: string,
): Promise<RemoteModel> {
  const worker = gpuWorker(assets);
  const absolute = new URL(url, location.href).href;
  const reply = await worker.ask({
    kind: "open",
    url: absolute,
    wasm: worker.wasm,
  });
  if (reply.kind === "failed") {
    throw new Error(reply.reason);
  }
  if (reply.kind !== "opened") {
    throw new Error(`unexpected reply ${reply.kind} while opening ${url}`);
  }
  return {
    backend: reply.backend,
    gpuFailure: reply.gpuFailure,
    url: absolute,
    worker,
    run: async (input, dims) => {
      const ran = await worker.ask({ kind: "run", url: absolute, input, dims });
      return ran.kind === "ran" ? ran.output : null;
    },
  };
}
