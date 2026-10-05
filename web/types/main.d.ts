import { type KeyOutline } from "./keycore";
declare global {
    interface Window {
        pianocvDrawnKeys?: readonly KeyOutline[];
    }
}
