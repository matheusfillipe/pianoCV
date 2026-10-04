import { BoxGeometry, Mesh, PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { cornersVisible } from "./keybed3d";

const camera = new PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0, 10);
camera.lookAt(0, 0, 0);
camera.updateMatrixWorld();

const wall = new Mesh(new BoxGeometry(4, 4, 1));
wall.position.set(0, 0, 5);
wall.updateMatrixWorld();

describe("cornersVisible", () => {
  const corners = [
    new Vector3(0, 0, 0),
    new Vector3(0, 0, 8),
    new Vector3(50, 0, 0),
  ];

  it("hides a corner behind an occluder or outside the frustum", () => {
    expect(cornersVisible(camera, corners, [wall])).toEqual([
      false,
      true,
      false,
    ]);
  });

  it("ignores an occluder that is not visible", () => {
    wall.visible = false;
    expect(cornersVisible(camera, corners, [wall])[0]).toBe(true);
    wall.visible = true;
  });
});
