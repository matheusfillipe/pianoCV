import { Group, Raycaster } from "three";
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
    setMaskMode(enabled: boolean): void;
    pick(camera: Parameters<Raycaster["setFromCamera"]>[1], x: number, y: number): KeyIdSpec | null;
}
export declare function instanceMaskColor(index: number): number;
export declare function keyIdSpecs(): KeyIdSpec[];
export declare function createKeyIdOverlay(): KeyIdOverlay;
