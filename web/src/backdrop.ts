import { CanvasTexture, RepeatWrapping, SRGBColorSpace } from "three";

const SIZE = 512;

export type BackdropKind = "clutter" | "noise" | "gradient";

function context(): CanvasRenderingContext2D {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2d canvas context unavailable");
  }
  return ctx;
}

function clutter(random: () => number): HTMLCanvasElement {
  const ctx = context();
  ctx.fillStyle = `hsl(${random() * 360},${20 + random() * 30}%,${20 + random() * 50}%)`;
  ctx.fillRect(0, 0, SIZE, SIZE);
  for (let i = 0; i < 14; i += 1) {
    ctx.fillStyle = `hsl(${random() * 360},${random() * 40}%,${random() * 80}%)`;
    ctx.fillRect(
      random() * SIZE,
      random() * SIZE,
      random() * SIZE * 0.6,
      random() * SIZE * 0.6,
    );
  }
  return ctx.canvas;
}

function noise(random: () => number): HTMLCanvasElement {
  const ctx = context();
  const image = ctx.createImageData(SIZE, SIZE);
  const scale = 1 + Math.floor(random() * 12);
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const cell = Math.floor(x / scale) * 73 + Math.floor(y / scale) * 151;
      const value = ((Math.sin(cell) + 1) / 2) * 255;
      const index = (y * SIZE + x) * 4;
      image.data[index] = value;
      image.data[index + 1] = value;
      image.data[index + 2] = value;
      image.data[index + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  return ctx.canvas;
}

function gradient(random: () => number): HTMLCanvasElement {
  const ctx = context();
  const ramp = ctx.createLinearGradient(0, 0, SIZE * random(), SIZE);
  ramp.addColorStop(0, `hsl(${random() * 360},40%,${10 + random() * 40}%)`);
  ramp.addColorStop(1, `hsl(${random() * 360},40%,${10 + random() * 60}%)`);
  ctx.fillStyle = ramp;
  ctx.fillRect(0, 0, SIZE, SIZE);
  return ctx.canvas;
}

const FLOOR_KINDS = ["planks", "tiles", "carpet", "plain"] as const;

function floorCanvas(random: () => number): HTMLCanvasElement {
  const ctx = context();
  const kind = FLOOR_KINDS[Math.floor(random() * FLOOR_KINDS.length)];
  const hue = random() * 360;
  const saturation = random() * 45;
  const lightness = 15 + random() * 65;
  const tone = (shift: number): string =>
    `hsl(${hue},${saturation}%,${Math.min(95, Math.max(5, lightness + shift))}%)`;
  ctx.fillStyle = tone(0);
  ctx.fillRect(0, 0, SIZE, SIZE);
  if (kind === "planks") {
    const plankWidth = SIZE / (3 + Math.floor(random() * 5));
    for (let y = 0; y < SIZE; y += plankWidth) {
      for (
        let x = -random() * SIZE;
        x < SIZE;
        x += SIZE * (0.4 + random() * 0.6)
      ) {
        ctx.fillStyle = tone((random() - 0.5) * 18);
        ctx.fillRect(x, y, SIZE, plankWidth);
        ctx.fillStyle = tone(-22);
        ctx.fillRect(x, y, 2, plankWidth);
      }
      ctx.fillStyle = tone(-22);
      ctx.fillRect(0, y, SIZE, 2);
    }
  } else if (kind === "tiles") {
    const tile = SIZE / (2 + Math.floor(random() * 5));
    for (let y = 0; y < SIZE; y += tile) {
      for (let x = 0; x < SIZE; x += tile) {
        ctx.fillStyle = tone((random() - 0.5) * 8);
        ctx.fillRect(x, y, tile, tile);
      }
    }
    ctx.strokeStyle = tone(-25);
    ctx.lineWidth = 3;
    for (let i = 0; i <= SIZE; i += tile) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, SIZE);
      ctx.moveTo(0, i);
      ctx.lineTo(SIZE, i);
      ctx.stroke();
    }
  } else if (kind === "carpet") {
    for (let i = 0; i < 6000; i += 1) {
      ctx.fillStyle = tone((random() - 0.5) * 30);
      ctx.fillRect(random() * SIZE, random() * SIZE, 2, 2);
    }
  }
  return ctx.canvas;
}

export function makeFloorTexture(random: () => number): CanvasTexture {
  const texture = new CanvasTexture(floorCanvas(random));
  texture.colorSpace = SRGBColorSpace;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.repeat.set(25, 25);
  texture.anisotropy = 8;
  return texture;
}

export function makeBackdrop(
  kind: BackdropKind,
  random: () => number,
): CanvasTexture {
  const canvas =
    kind === "noise"
      ? noise(random)
      : kind === "gradient"
        ? gradient(random)
        : clutter(random);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}
