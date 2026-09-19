export interface HudState {
    live: boolean;
    corners: boolean;
    hands: boolean;
    input: boolean;
    label: boolean;
}
export interface Hud {
    readonly state: HudState;
    readonly capture: HTMLElement;
    status(key: string, value: string): void;
    onRedetect(handler: () => void): void;
    onAdopt(handler: () => void): void;
    onFlip(handler: () => void): void;
    onMeasure(handler: () => void): void;
}
export declare function styleButton(button: HTMLElement, active?: boolean): void;
export declare function createHud(): Hud;
