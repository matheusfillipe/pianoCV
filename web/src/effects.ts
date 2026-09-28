import { outsideBlackKeys, tracePolygon } from "./draw";
import type { Point } from "./homography";
import { STANDARD_BOARDS } from "./keystrip";

declare global {
  interface Window {
    // lets a headless check play notes with no MIDI hardware attached
    pianocvPlay?: (pitch: number, velocity: number) => void;
  }
}

/** The lowest MIDI pitch of a standard board with this many white keys starting on this
 * letter, or null when it does not match any standard size. */
export function lowestPitchFor(
  whiteKeys: number,
  phase: string,
): number | null {
  const board = STANDARD_BOARDS.find(
    (b) => b.whiteKeys === whiteKeys && b.phase === phase,
  );
  return board?.lowestPitch ?? null;
}

type NoteState = {
  readonly velocity: number;
  readonly releasedAt: number | null;
};

export type EffectsState = {
  readonly notes: ReadonlyMap<number, NoteState>;
};

const FADE_MS = 400;

export function createEffectsState(): EffectsState {
  return { notes: new Map() };
}

export function noteOn(
  state: EffectsState,
  pitch: number,
  velocity: number,
  now: number,
): EffectsState {
  if (velocity <= 0) {
    return noteOff(state, pitch, now);
  }
  const notes = new Map(state.notes);
  notes.set(pitch, { velocity, releasedAt: null });
  return { notes };
}

export function noteOff(
  state: EffectsState,
  pitch: number,
  now: number,
): EffectsState {
  const existing = state.notes.get(pitch);
  if (!existing || existing.releasedAt !== null) {
    return state;
  }
  const notes = new Map(state.notes);
  notes.set(pitch, { ...existing, releasedAt: now });
  return { notes };
}

/** How lit a pitch is right now: full from its velocity while held, fading linearly to
 * nothing over FADE_MS once released. */
export function glowLevel(
  state: EffectsState,
  pitch: number,
  now: number,
): number {
  const note = state.notes.get(pitch);
  if (!note) {
    return 0;
  }
  const base = note.velocity / 127;
  if (note.releasedAt === null) {
    return base;
  }
  const elapsed = now - note.releasedAt;
  if (elapsed >= FADE_MS) {
    return 0;
  }
  return base * (1 - elapsed / FADE_MS);
}

/** Drops notes that have fully faded, so the map does not grow for as long as the page runs. */
export function prune(state: EffectsState, now: number): EffectsState {
  let changed = false;
  const notes = new Map(state.notes);
  for (const [pitch, note] of state.notes) {
    if (note.releasedAt !== null && now - note.releasedAt >= FADE_MS) {
      notes.delete(pitch);
      changed = true;
    }
  }
  return changed ? { notes } : state;
}

function connectMidi(play: (pitch: number, velocity: number) => void): void {
  if (typeof navigator.requestMIDIAccess !== "function") {
    return;
  }
  const onMessage = (event: MIDIMessageEvent): void => {
    const data = event.data;
    if (data === null || data.length < 3) {
      return;
    }
    const status = data[0] & 0xf0;
    if (status === 0x90) {
      play(data[1], data[2]);
    } else if (status === 0x80) {
      play(data[1], 0);
    }
  };
  const attachAll = (access: MIDIAccess): void => {
    access.inputs.forEach((input) => {
      input.onmidimessage = onMessage;
    });
  };
  navigator.requestMIDIAccess().then(
    (access) => {
      attachAll(access);
      access.onstatechange = () => attachAll(access);
    },
    () => {
      // MIDI unavailable or refused; window.pianocvPlay still drives the effect
    },
  );
}

const GLOW_COLOR = "255, 196, 92";

export type Effects = {
  readonly draw: (
    ctx: CanvasRenderingContext2D,
    faces: readonly {
      readonly black: boolean;
      readonly semitone: number;
      readonly bar: readonly Point[];
    }[],
    lowestPitch: number | null,
    w: number,
    h: number,
    now: number,
  ) => void;
};

export function createEffects(): Effects {
  let state = createEffectsState();
  const play = (pitch: number, velocity: number): void => {
    state =
      velocity > 0
        ? noteOn(state, pitch, velocity, performance.now())
        : noteOff(state, pitch, performance.now());
  };
  window.pianocvPlay = play;
  connectMidi(play);

  return {
    draw: (ctx, faces, lowestPitch, w, h, now) => {
      state = prune(state, now);
      if (lowestPitch === null) {
        return;
      }
      ctx.fillStyle = `rgba(${GLOW_COLOR}, 1)`;
      const glow = (black: boolean): void => {
        for (const face of faces) {
          const level = glowLevel(state, lowestPitch + face.semitone, now);
          if (face.black !== black || level <= 0) {
            continue;
          }
          ctx.globalAlpha = level;
          ctx.beginPath();
          tracePolygon(ctx, face.bar, w, h);
          ctx.fill();
        }
      };
      outsideBlackKeys(ctx, faces, w, h, () => glow(false));
      glow(true);
      ctx.globalAlpha = 1;
    },
  };
}
