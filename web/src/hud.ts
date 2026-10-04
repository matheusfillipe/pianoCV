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

const PANEL_STYLE: Partial<CSSStyleDeclaration> = {
  position: "fixed",
  top: "12px",
  left: "12px",
  zIndex: "10",
  display: "flex",
  flexDirection: "column",
  gap: "6px",
  padding: "10px 12px",
  minWidth: "270px",
  background: "rgba(8,8,10,0.72)",
  backdropFilter: "blur(6px)",
  border: "1px solid rgba(229,229,229,0.12)",
  borderRadius: "10px",
  color: "#e5e5e5",
  font: "12px/1.5 ui-sans-serif, system-ui, sans-serif",
};

export function styleButton(button: HTMLElement, active = false): void {
  Object.assign(button.style, {
    background: active ? "rgba(56,189,248,0.22)" : "rgba(255,255,255,0.05)",
    border: `1px solid ${active ? "rgba(56,189,248,0.65)" : "rgba(229,229,229,0.18)"}`,
    borderRadius: "6px",
    color: active ? "#bae6fd" : "#d4d4d4",
    cursor: "pointer",
    font: "inherit",
    padding: "3px 9px",
  });
}

function row(parent: HTMLElement, label: string): HTMLElement {
  const line = document.createElement("div");
  const name = document.createElement("span");
  const controls = document.createElement("div");
  name.textContent = label;
  Object.assign(line.style, {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  });
  Object.assign(name.style, {
    width: "52px",
    color: "#8a8a8a",
    flex: "0 0 auto",
  });
  Object.assign(controls.style, {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    flexWrap: "wrap",
  });
  line.appendChild(name);
  line.appendChild(controls);
  parent.appendChild(line);
  return controls;
}

export function createHud(): Hud {
  const state: HudState = {
    corners: false,
    hands: false,
    keys: false,
    glow: false,
  };
  const panel = document.createElement("div");
  Object.assign(panel.style, PANEL_STYLE);

  const viewRow = row(panel, "show");
  const capture = row(panel, "capture");

  const readouts = document.createElement("div");
  Object.assign(readouts.style, {
    display: "grid",
    gridTemplateColumns: "52px 1fr",
    gap: "0 8px",
    marginTop: "2px",
    paddingTop: "6px",
    borderTop: "1px solid rgba(229,229,229,0.1)",
    fontVariantNumeric: "tabular-nums",
  });
  panel.appendChild(readouts);

  const values = new Map<string, HTMLElement>();
  const status = (key: string, value: string): void => {
    let cell = values.get(key);
    if (!cell) {
      const name = document.createElement("span");
      name.textContent = key;
      name.style.color = "#8a8a8a";
      cell = document.createElement("span");
      readouts.appendChild(name);
      readouts.appendChild(cell);
      values.set(key, cell);
    }
    cell.textContent = value;
  };

  const refresh: (() => void)[] = [];
  const toggle = (
    parent: HTMLElement,
    text: string,
    title: string,
    read: () => boolean,
    write: (on: boolean) => void,
  ): void => {
    const button = document.createElement("button");
    button.textContent = text;
    button.title = title;
    const paint = (): void => styleButton(button, read());
    paint();
    refresh.push(paint);
    button.addEventListener("click", () => {
      write(!read());
      for (const fn of refresh) {
        fn();
      }
    });
    parent.appendChild(button);
  };

  toggle(
    viewRow,
    "manual",
    "drag the four corners onto the keybed yourself",
    () => state.corners,
    (on) => {
      state.corners = on;
    },
  );
  toggle(
    viewRow,
    "hands",
    "overlay mediapipe hand landmarks",
    () => state.hands,
    (on) => {
      state.hands = on;
    },
  );
  toggle(
    viewRow,
    "keys",
    "draw the detected keys over the video",
    () => state.keys,
    (on) => {
      state.keys = on;
    },
  );
  toggle(
    viewRow,
    "glow",
    "light the key each note plays",
    () => state.glow,
    (on) => {
      state.glow = on;
    },
  );

  const flip = document.createElement("button");
  flip.textContent = "flip";
  flip.title =
    "swap which edge is the back, when the corners sit right but read reversed";
  styleButton(flip);
  capture.appendChild(flip);

  const shortcuts: Record<string, () => void> = {
    c: () => {
      state.corners = !state.corners;
    },
    h: () => {
      state.hands = !state.hands;
    },
    k: () => {
      state.keys = !state.keys;
    },
    g: () => {
      state.glow = !state.glow;
    },
  };
  window.addEventListener("keydown", (event) => {
    const action = shortcuts[event.key];
    if (!action || event.repeat) {
      return;
    }
    action();
    for (const fn of refresh) {
      fn();
    }
  });

  document.body.appendChild(panel);
  return {
    state,
    capture,
    status,
    onFlip: (handler) => flip.addEventListener("click", handler),
  };
}
