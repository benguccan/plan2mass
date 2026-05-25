from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)


def _setup_world() -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.76, 0.82, 0.90, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def _import_model(path: Path) -> None:
    if path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif path.suffix.lower() == ".gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise RuntimeError(f"Unsupported model format: {path.suffix}")


def _get_bounds():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found after import.")

    min_x = min((o.bound_box[i][0] + o.location.x) for o in meshes for i in range(8))
    max_x = max((o.bound_box[i][0] + o.location.x) for o in meshes for i in range(8))
    min_y = min((o.bound_box[i][1] + o.location.y) for o in meshes for i in range(8))
    max_y = max((o.bound_box[i][1] + o.location.y) for o in meshes for i in range(8))
    min_z = min((o.bound_box[i][2] + o.location.z) for o in meshes for i in range(8))
    max_z = max((o.bound_box[i][2] + o.location.z) for o in meshes for i in range(8))

    return (min_x, max_x, min_y, max_y, min_z, max_z)


def _setup_camera_and_lights() -> None:
    min_x, max_x, min_y, max_y, min_z, max_z = _get_bounds()
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    cz = (min_z + max_z) * 0.5
    span = max(max_x - min_x, max_y - min_y, (max_z - min_z)) or 1.0

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    dist = span * 2.3
    cam_obj.location = (cx + dist, cy - dist * 0.9, cz + span * 1.2)
    direction = (cx - cam_obj.location.x, cy - cam_obj.location.y, cz - cam_obj.location.z)
    rot = direction_to_euler(direction)
    cam_obj.rotation_euler = rot

    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 4.5
    sun = bpy.data.objects.new(name="Sun", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (cx + span, cy - span, cz + span * 2.0)
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(35))

    fill_data = bpy.data.lights.new(name="Fill", type="AREA")
    fill_data.energy = 500
    fill_data.size = span * 0.8
    fill = bpy.data.objects.new(name="Fill", object_data=fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.location = (cx - span * 0.8, cy + span * 0.6, cz + span * 0.7)


def direction_to_euler(direction):
    dx, dy, dz = direction
    dist_xy = math.sqrt(dx * dx + dy * dy) or 1e-6
    pitch = math.atan2(-dz, dist_xy)
    yaw = math.atan2(dx, dy)
    return (pitch, 0.0, yaw)


def main() -> int:
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender -b <file> -P render_script.py -- <model_path> <output_path>")
        return 2

    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        print("Missing args. Need: <model_path> <output_path>")
        return 2

    model_path = Path(args[0]).resolve()
    output_path = Path(args[1]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 2

    _clear_scene()
    _setup_world()
    _import_model(model_path)
    _setup_camera_and_lights()

    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

