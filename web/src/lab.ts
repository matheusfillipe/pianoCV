import type { Corners } from "./calibrate";
import { styleButton } from "./hud";
import { checkQuad } from "./quad";

interface Sidecar {
  kind: "rec" | "snap";
  startedAt: number;
  durationMs: number;
  corners: Corners | null;
  imageWidth: number;
  imageHeight: number;
  mimeType: string;
}

interface LabBar {
  rec: HTMLButtonElement;
  snap: HTMLButtonElement;
  timer: HTMLSpanElement;
  status: HTMLSpanElement;
}

export interface LabOptions {
  video: HTMLVideoElement;
  stream: MediaStream;
  mount: HTMLElement;
  getCorners(): Corners | null;
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function formatTimer(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  return `${pad2(Math.floor(totalSeconds / 60))}:${pad2(totalSeconds % 60)}`;
}

function fileStamp(date: Date): string {
  return `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}-${pad2(date.getHours())}${pad2(date.getMinutes())}${pad2(date.getSeconds())}`;
}

function pickMimeType(): string | undefined {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8"];
  for (const candidate of candidates) {
    if (MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

function containerExtension(mimeType: string): string {
  const subtype = mimeType.split(";")[0].split("/")[1];
  return subtype || "webm";
}

function createBar(mount: HTMLElement): LabBar {
  const rec = document.createElement("button");
  const snap = document.createElement("button");
  const timer = document.createElement("span");
  const status = document.createElement("span");
  rec.textContent = "rec";
  snap.textContent = "snap";
  timer.textContent = "00:00";
  for (const button of [rec, snap]) {
    styleButton(button);
    mount.appendChild(button);
  }
  timer.style.fontVariantNumeric = "tabular-nums";
  status.style.color = "#8a8a8a";
  mount.appendChild(timer);
  mount.appendChild(status);
  return { rec, snap, timer, status };
}

function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function postToLab(
  name: string,
  body: Blob | string,
  contentType?: string,
): Promise<void> {
  const response = await fetch(`/lab/save/${name}`, {
    method: "POST",
    body,
    headers: contentType ? { "Content-Type": contentType } : undefined,
  });
  if (!response.ok) {
    throw new Error(`lab save failed: ${response.status}`);
  }
}

/** Draws one video frame onto a same-sized canvas and reads it back as a PNG blob, or null while
 * the video has no dimensions yet. */
export function captureFrame(video: HTMLVideoElement): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  if (canvas.width === 0 || canvas.height === 0) {
    return Promise.resolve(null);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return Promise.resolve(null);
  }
  ctx.drawImage(video, 0, 0);
  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/png");
  });
}

export function createLab(options: LabOptions): void {
  const { video, stream, getCorners, mount } = options;
  const bar = createBar(mount);
  let recorder: MediaRecorder | null = null;
  let startedAt = 0;
  let timerInterval: number | null = null;
  let flashTimeout: number | null = null;

  const flash = (text: string): void => {
    bar.status.textContent = text;
    if (flashTimeout !== null) {
      window.clearTimeout(flashTimeout);
    }
    flashTimeout = window.setTimeout(() => {
      bar.status.textContent = "";
      flashTimeout = null;
    }, 1500);
  };

  const setTimer = (active: boolean): void => {
    if (timerInterval !== null) {
      window.clearInterval(timerInterval);
      timerInterval = null;
    }
    bar.timer.textContent = "00:00";
    if (active) {
      timerInterval = window.setInterval(() => {
        bar.timer.textContent = formatTimer(Date.now() - startedAt);
      }, 500);
    }
  };

  const save = (name: string, blob: Blob, sidecar: Sidecar): void => {
    const stem = name.split(".")[0];
    postToLab(name, blob)
      .then(() =>
        postToLab(`${stem}.json`, JSON.stringify(sidecar), "application/json"),
      )
      .then(() => {
        flash("saved");
      })
      .catch(() => {
        download(blob, name);
        flash("download fallback");
      });
  };

  const guard = (): boolean => {
    const corners = getCorners();
    const check = corners
      ? checkQuad(corners)
      : { usable: false, reason: "no corners" };
    if (!check.usable) {
      flash(`not saved: ${check.reason}`);
    }
    return check.usable;
  };

  const snapshot = (): void => {
    if (!guard()) {
      return;
    }
    const capturedAt = Date.now();
    void captureFrame(video).then((blob) => {
      if (!blob) {
        return;
      }
      save(`snap-${fileStamp(new Date())}.png`, blob, {
        kind: "snap",
        startedAt: capturedAt,
        durationMs: 0,
        corners: getCorners(),
        imageWidth: video.videoWidth,
        imageHeight: video.videoHeight,
        mimeType: "image/png",
      });
    });
  };

  const toggleRecording = (): void => {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      return;
    }
    const mimeType = pickMimeType();
    const next = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    next.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };
    startedAt = Date.now();
    next.onstop = () => {
      const durationMs = Date.now() - startedAt;
      const corners = getCorners();
      const blob = new Blob(chunks, { type: next.mimeType });
      recorder = null;
      styleButton(bar.rec);
      setTimer(false);
      save(
        `rec-${fileStamp(new Date())}.${containerExtension(next.mimeType)}`,
        blob,
        {
          kind: "rec",
          startedAt,
          durationMs,
          corners: corners && checkQuad(corners).usable ? corners : null,
          imageWidth: video.videoWidth,
          imageHeight: video.videoHeight,
          mimeType: next.mimeType,
        },
      );
    };
    next.start();
    recorder = next;
    styleButton(bar.rec);
    bar.rec.style.background = "#dc2626";
    bar.rec.style.borderColor = "#dc2626";
    setTimer(true);
  };

  bar.rec.addEventListener("click", toggleRecording);
  bar.snap.addEventListener("click", snapshot);
  window.addEventListener("keydown", (event) => {
    if (event.repeat) {
      return;
    }
    if (event.key === "r") {
      toggleRecording();
    }
    if (event.key === "s") {
      snapshot();
    }
  });
}
