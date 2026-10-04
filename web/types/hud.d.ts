export interface HudState {
    corners: boolean;
    hands: boolean;
    keys: boolean;
    glow: boolean;
}
export interface Hud {
    readonly state: HudState;
    readonly capture: HTMLElement;
    status(key: string, value: string): void;
    onFlip(handler: () => void): void;
}
export declare function styleButton(button: HTMLElement, active?: boolean): void;
export declare function createHud(): Hud;
