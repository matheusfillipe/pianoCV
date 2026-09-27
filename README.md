# pianoCV

A set of small computer vision models, and the browser pipeline around them, that finds every
key of a piano or electronic keyboard in a camera image, live, and solves where the keyboard
sits in 3D.

Everything runs in the browser. There is no server and nothing you record leaves your machine.
It never listens to the music and never tries to work out which notes are played: it solves
geometry only, and what you get back is every key's outline in the camera's own perspective,
ready to draw anything onto. Play a MIDI keyboard into the page and the keys you press light
up where they are in the picture.

## Run it

```
make install
make model
make dev
```

Then open http://localhost:5273/ and allow the camera. `make model` downloads the trained
weights from Hugging Face, which are not stored in this repo. `make help` lists everything
else.

The key segmenter runs on the GPU through WebGPU where the browser has it, and on the CPU
through WebAssembly where it does not.

## How it finds the keys

Three models work together, and each one covers what the others are bad at.

1. **The keybed detector** finds roughly where the keyboard is. It is a MobileNetV3 U-Net that
   marks keyboard pixels in a 288 x 288 frame, and the pipeline holds the region it finds
   steady from frame to frame.
2. **The key segmenter** looks at a rotated crop around that region and marks, for every pixel,
   whether it is a white key, a black key, the gap between two white keys, or neither. The
   outline of the keyboard comes from these key pixels, so a rough first region never bends
   the keys.
3. **The key matcher** reads the white-key gaps and the black-key edges along the keyboard, at
   two depths. Where the gaps lean, the outline was skewed, so the pipeline straightens the
   outline to the key lines and reads again.

A real keyboard is not any arrangement of keys: it comes in a handful of sizes and always
repeats the same pattern of two and three black keys. So the pipeline fits that pattern to
what the models saw, which settles how many keys there are, which note each one is, and how
high the black keys stand. Every key is always there and never drawn twice. Each key is then
drawn with the shape the segmenter saw for it, black key sides included, and keeps the fitted
shape wherever the segmenter could not see it cleanly.

## Train it

All three models are trained on synthetic renders, where every key's outline is known
exactly. The generator builds a procedural keyboard with real proportions, varies its size,
case, lighting, background and camera, and saves each frame with per-key labels.

```
PIANOCV_GEN_OUT=synth-keys make lab-synth-generate
make lab-keyseg-train
make lab-keymatch-train
```

The first command renders 8000 frames into `data/synth-keys/`, which takes about an hour.
`PIANOCV_GEN_POSE="elevation=12,40&azimuth=45,110"` narrows the camera angles when you want
more of a particular view. Each trainer writes a `.pt` checkpoint and a `.onnx` model under
`data/models/`. Copy the `.onnx` files into `web/public/` and reload the page to run them. A
GPU makes training take minutes instead of hours.

`make lab-keyseg-train ARGS="--preview 12"` draws the labels over a dozen crops without
training, which is the quickest way to check them.

The keybed detector trains with `make lab-trainseg2`, and the notebooks under `tools/kaggle/`
train it on a hosted GPU.

## Check it

`make lab-keys-eval` runs the whole pipeline headless over every recording in
`data/recordings/` and over synthetic renders, and reports the board it read, how sure it was,
and how far each drawn key lands from the real one.

## The models

The weights live at [mattf/pianoCV](https://huggingface.co/mattf/pianoCV) rather than in this
repo, and `make model` fetches them into `web/public/`.

## Notes on the data

Recordings stay on your machine. Everything under `data/` is ignored by git on purpose,
because those files are pictures of your room.

## Licence

Apache 2.0.
