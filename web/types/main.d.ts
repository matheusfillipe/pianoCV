import type { Point } from "./homography";
declare global {
    interface Window {
        pianocvDrawnKeys?: readonly DrawnKey[];
    }
}
type DrawnKey = {
    readonly black: boolean;
    readonly semitone: number;
    readonly bar: readonly Point[];
};
export {};
