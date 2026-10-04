import { type KeyOutline } from "./keyoutlines";
declare global {
    interface Window {
        pianocvDrawnKeys?: readonly KeyOutline[];
    }
}
