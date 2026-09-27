export interface HudState {
  live: boolean;
  corners: boolean;
  hands: boolean;
  input: boolean;
  label: boolean;
  keys: boolean;
  glow: boolean;
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
  // corners stay on: they are the label that capture saves, and a hidden label is a wrong label
  // auto-labelling stays off: it writes pictures of whatever the camera sees to disk
  const state: HudState = {
    live: true,
    corners: true,
    hands: false,
    input: false,
    label: false,
    keys: false,
    glow: false,
  };
  const panel = document.createElement("div");
  Object.assign(panel.style, PANEL_STYLE);

  const detectRow = row(panel, "detect");
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
    detectRow,
    "live",
    "keep detecting every frame",
    () => state.live,
    (on) => {
      state.live = on;
    },
  );
  toggle(
    detectRow,
    "manual",
    "drag the four corners onto the keybed yourself",
    () => state.corners,
    (on) => {
      state.corners = on;
    },
  );
  toggle(
    detectRow,
    "hands",
    "overlay mediapipe hand landmarks",
    () => state.hands,
    (on) => {
      state.hands = on;
    },
  );
  toggle(
    detectRow,
    "input",
    "show the image the model actually sees",
    () => state.input,
    (on) => {
      state.input = on;
    },
  );
  toggle(
    detectRow,
    "keys",
    "draw the detected keys over the video",
    () => state.keys,
    (on) => {
      state.keys = on;
    },
  );
  toggle(
    detectRow,
    "glow",
    "light the key each note plays",
    () => state.glow,
    (on) => {
      state.glow = on;
    },
  );

  toggle(
    capture,
    "label",
    "save held frames as training labels",
    () => state.label,
    (on) => {
      state.label = on;
    },
  );

  const redetect = document.createElement("button");
  redetect.textContent = "detect again";
  redetect.title = "give up on the current keybed and start hunting again";
  styleButton(redetect);
  detectRow.appendChild(redetect);

  const adopt = document.createElement("button");
  adopt.textContent = "adopt";
  adopt.title = "copy the held rectangle into the draggable corners";
  styleButton(adopt);
  capture.appendChild(adopt);

  const flip = document.createElement("button");
  flip.textContent = "flip";
  flip.title =
    "swap which edge is the back, when the corners sit right but read reversed";
  styleButton(flip);
  capture.appendChild(flip);

  const measure = document.createElement("button");
  measure.textContent = "hold";
  measure.title =
    "hold the dragged corners as the keybed, measuring its shape or lens along the way";
  styleButton(measure);
  capture.appendChild(measure);

  const shortcuts: Record<string, () => void> = {
    l: () => {
      state.live = !state.live;
    },
    c: () => {
      state.corners = !state.corners;
    },
    h: () => {
      state.hands = !state.hands;
    },
    i: () => {
      state.input = !state.input;
    },
    a: () => {
      state.label = !state.label;
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
    onRedetect: (handler) => redetect.addEventListener("click", handler),
    onAdopt: (handler) => adopt.addEventListener("click", handler),
    onFlip: (handler) => flip.addEventListener("click", handler),
    onMeasure: (handler) => measure.addEventListener("click", handler),
  };
}
