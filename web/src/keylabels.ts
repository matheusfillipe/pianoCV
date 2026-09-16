import {
  BoxGeometry,
  Color,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  Raycaster,
  Vector2,
} from "three";
import { BACK_X, FRONT_X, KEY_TOP_Y, SPAN, SPAN_MIN_Z } from "./keybed3d";
import { isBlack, keyRect } from "./keys";

const KEY_BOTTOM_Y = 0.0476;
const WHITE_TOP_Y = KEY_TOP_Y;
const BLACK_TOP_Y = 0.8902;
const BLACK_FRONT_X = 0.4204;
const MODEL_LOW_PITCH = 17;
const MODEL_HIGH_PITCH = 112;
const MODEL_WHITE_COUNT = 56;
const MODEL_FIRST_KEY = keyRect(MODEL_LOW_PITCH);
const BLACK_KEY_LENGTH = 0.305;
const WHITE_GAP = 0.018;
const BLACK_GAP = 0.012;
const raycaster = new Raycaster();
const pointer = new Vector2();

export interface KeyIdSpec {
  pitch: number;
  label: string;
  black: boolean;
  u0: number;
  u1: number;
  v1: number;
}

export interface KeyIdOverlay {
  group: Group;
  pick(
    camera: Parameters<Raycaster["setFromCamera"]>[1],
    x: number,
    y: number,
  ): KeyIdSpec | null;
}

function name(pitch: number): string {
  const names = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
  ];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

export function keyIdSpecs(): KeyIdSpec[] {
  const specs: KeyIdSpec[] = [];
  for (let pitch = MODEL_LOW_PITCH; pitch <= MODEL_HIGH_PITCH; pitch += 1) {
    const rect = keyRect(pitch);
    const u0 = ((rect.u0 - MODEL_FIRST_KEY.u0) * 36) / MODEL_WHITE_COUNT;
    const u1 = isBlack(pitch)
      ? u0 + BLACK_KEY_LENGTH / SPAN
      : ((rect.u1 - MODEL_FIRST_KEY.u0) * 36) / MODEL_WHITE_COUNT;
    specs.push({
      pitch,
      label: name(pitch),
      black: isBlack(pitch),
      u0,
      u1,
      v1: isBlack(pitch) ? (BLACK_FRONT_X - BACK_X) / (FRONT_X - BACK_X) : 1,
    });
  }
  return specs;
}

export function createKeyIdOverlay(): KeyIdOverlay {
  const group = new Group();
  group.name = "key-id-overlay";
  const specs = keyIdSpecs();
  const byUuid = new Map<string, KeyIdSpec>();
  const meshes: Mesh[] = [];
  for (const [index, spec] of specs.entries()) {
    const front = spec.black ? BLACK_FRONT_X : FRONT_X;
    const width = front - BACK_X;
    const length = (spec.u1 - spec.u0) * SPAN;
    const height = (spec.black ? BLACK_TOP_Y : WHITE_TOP_Y) - KEY_BOTTOM_Y;
    const gap = spec.black ? BLACK_GAP : WHITE_GAP;
    const color = new Color().setHSL((index * 0.61803398875) % 1, 0.82, 0.58);
    const material = new MeshBasicMaterial({ color });
    const geometry = new BoxGeometry(
      width,
      height,
      Math.max(length - gap, gap),
    );
    const mesh = new Mesh(geometry, material);
    mesh.position.set(
      BACK_X + width / 2,
      KEY_BOTTOM_Y + height / 2,
      SPAN_MIN_Z + ((spec.u0 + spec.u1) * SPAN) / 2,
    );
    byUuid.set(mesh.uuid, spec);
    meshes.push(mesh);
    group.add(mesh);
    const outline = new LineSegments(
      new EdgesGeometry(geometry),
      new LineBasicMaterial({ color: 0x080808 }),
    );
    outline.position.copy(mesh.position);
    group.add(outline);
  }
  group.visible = false;
  return {
    group,
    pick(camera, x, y) {
      pointer.set(x, y);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(meshes, false)[0];
      return hit ? (byUuid.get(hit.object.uuid) ?? null) : null;
    },
  };
}
