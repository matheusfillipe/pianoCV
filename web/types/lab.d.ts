import type { Corners } from "./calibrate";
export interface LabOptions {
    video: HTMLVideoElement;
    stream: MediaStream;
    mount: HTMLElement;
    getCorners(): Corners | null;
}
export declare function postToLab(name: string, body: Blob | string, contentType?: string): Promise<void>;
/** Draws one video frame onto a same-sized canvas and reads it back as a PNG blob, or null while
 * the video has no dimensions yet. */
export declare function captureFrame(video: HTMLVideoElement): Promise<Blob | null>;
export declare function createLab(options: LabOptions): void;
