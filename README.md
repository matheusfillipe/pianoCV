# pianoCV

Point a webcam at a piano and pianoCV finds every key, live, right in your browser.

**[Try it here](https://matheusfillipe.github.io/pianoCV/)**. Nothing gets uploaded: the
camera never leaves your machine.

Plug in a MIDI keyboard and the keys you play light up where they really are in the picture.
The idea is to draw visuals onto a real piano, like a video filter.

## How it works

KeyNet, one small keypoint model, looks at the camera and marks the keybed corners, the gaps
between white keys and the edges of the black keys. A small Rust core, compiled to
WebAssembly, fits a real keyboard layout to those points and tracks it, so every key is found,
even the blurry ones far away, and each one gets a note.

The model runs in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/),
on your GPU through [WebGPU](https://developer.mozilla.org/docs/Web/API/WebGPU_API) when it can.
It was trained with [PyTorch](https://pytorch.org) on thousands of fake pianos rendered in
[three.js](https://threejs.org) plus hand-checked real frames, and it lives on
[Hugging Face](https://huggingface.co/mattf/pianoCV). Your hands are found with
[MediaPipe](https://ai.google.dev/edge/mediapipe) so the glow stays behind them, and the
notes come in through [Web MIDI](https://developer.mozilla.org/docs/Web/API/Web_MIDI_API).

## Use it natively

The same Rust core that runs in the browser as WebAssembly can also run natively on your machine.
Feed it a camera frame in RGBA and it returns every key with its note and outline.
Try it on a video with:

```
make model
make core-demo ARGS="path/to/video.mp4 path/to/output"
```

This needs ffmpeg, the model from `make model`, and a Rust toolchain.

## The model

The trained weights are on Hugging Face at
[mattf/pianoCV](https://huggingface.co/mattf/pianoCV), with a model card that explains what
it takes and returns:
[`keynet.onnx`](https://huggingface.co/mattf/pianoCV/resolve/main/keynet.onnx).

It is a plain [ONNX](https://onnx.ai) file, so you can use it from any language with
[ONNX Runtime](https://onnxruntime.ai).

## Run it yourself

Install [Bun](https://bun.sh/docs/installation),
[uv](https://docs.astral.sh/uv/getting-started/installation/) and a
[Rust toolchain](https://rustup.rs) with [wasm-pack](https://rustwasm.github.io/wasm-pack/), then:

```
git clone https://github.com/matheusfillipe/pianoCV.git
cd pianoCV
make install
make model
make dev
```

`make install` sets up the browser app and the Python tools, `make model` downloads the model
from Hugging Face, and `make dev` starts the app. Open http://localhost:5273/, allow the
camera, and point it at a keyboard. `make help` shows everything else, including how to render
training pianos and train the model yourself.

## Licence

Apache 2.0.
