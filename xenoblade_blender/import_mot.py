import bpy
import time
import numpy as np

from xenoblade_blender.import_root import init_logging

from . import xc3_model_py
from .export_root import export_skeleton

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty


class ImportMot(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade animation"""

    bl_idname = "import_scene.mot"
    bl_label = "Import Mot"

    filename_ext = ".mot"

    filter_glob: StringProperty(
        default="*.mot;*.anm;*.motstm_data;*.sar",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        import_mot(self, context, self.filepath)
        return {"FINISHED"}


def import_mot(operator: bpy.types.Operator, context: bpy.types.Context, path: str):
    start = time.time()

    animations = xc3_model_py.load_animations(path)

    end = time.time()
    print(f"Load {len(animations)} Animations: {end - start}")

    start = time.time()

    if context.object is None or not isinstance(
        context.object.data, bpy.types.Armature
    ):
        operator.report({"ERROR"}, "No armature selected")
        return

    armature = context.object
    if armature.animation_data is None:
        armature.animation_data_create()

    skeleton = export_skeleton(armature)

    # Animations expect the in game ordering for bones.
    bone_names = {bone.name for bone in skeleton.bones}

    for i, bone in enumerate(skeleton.bones):
        if bone.parent_index is not None and bone.parent_index > i:
            print(f"invalid index {bone.parent_index} > {i}")

    # TODO: Is this the best way to load all animations?
    # TODO: Optimize this.
    for animation in animations:
        action = import_animation(armature, skeleton, bone_names, animation)
        armature.animation_data.action = action

        if bpy.app.version >= (4, 4, 0):
            # TODO: Why are there empty animations that don't create slots?
            if len(action.slots) == 0:
                slot = action.slots.new(id_type="OBJECT", name="Legacy Slot")
            else:
                slot = action.slots[0]

            # Automatic slot assignment works differently on 5.0 or later.
            armature.animation_data.action_slot = action.slots[0]

    end = time.time()
    print(f"Import Blender Animation: {end - start}")


def import_animation(armature, skeleton, bone_names, animation):
    action = bpy.data.actions.new(animation.name)
    if animation.frame_count > 0:
        action.frame_end = float(animation.frame_count) - 1.0

    # Reset between each animation.
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()

    fcurves = animation.fcurves(skeleton, use_blender_coordinates=True)
    locations = fcurves.translation
    rotations_xyzw = fcurves.rotation
    scales = fcurves.scale

    for name, values in locations.items():
        if name in bone_names:
            set_fcurves_component(action, name, "location", values[:, 0], 0)
            set_fcurves_component(action, name, "location", values[:, 1], 1)
            set_fcurves_component(action, name, "location", values[:, 2], 2)

    for name, values in rotations_xyzw.items():
        if name in bone_names:
            # Blender uses wxyz instead of xyzw.
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 3], 0)
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 0], 1)
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 1], 2)
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 2], 3)

    for name, values in scales.items():
        if name in bone_names:
            set_fcurves_component(action, name, "scale", values[:, 0], 0)
            set_fcurves_component(action, name, "scale", values[:, 1], 1)
            set_fcurves_component(action, name, "scale", values[:, 2], 2)

    return action


def set_fcurves_component(
    action, bone_name: str, value_name: str, values: np.ndarray, i: int
):
    # Values can be quickly set in the form [frame, value, frame, value, ...]
    # Assume one value at each frame index for now.
    keyframe_points = np.zeros((values.size, 2), dtype=np.float32)
    keyframe_points[:, 0] = np.arange(values.size)
    keyframe_points[:, 1] = values

    # Each coordinate of each value has its own fcurve.
    data_path = f'pose.bones["{bone_name}"].{value_name}'
    fcurve = create_fcurve(action, data_path, i, bone_name)
    fcurve.keyframe_points.add(count=values.size)
    fcurve.keyframe_points.foreach_set("co", keyframe_points.reshape(-1))


def create_fcurve(
    action, data_path: str, index: int, group_name: str
) -> bpy.types.FCurve:
    if bpy.app.version >= (5, 0, 0):
        # Blender 5.0 removes the legacy Action API.
        if len(action.layers) == 0:
            layer = action.layers.new("Layer")
        else:
            layer = action.layers[0]

        if len(layer.strips) == 0:
            strip = layer.strips.new(type="KEYFRAME")
        else:
            strip = layer.strips[0]

        if len(action.slots) == 0:
            slot = action.slots.new(id_type="OBJECT", name="Legacy Slot")
        else:
            slot = action.slots[0]

        channelbag = strip.channelbag(slot, ensure=True)
        return channelbag.fcurves.new(data_path, index=index, group_name=group_name)
    else:
        return action.fcurves.new(data_path, index=index, action_group=group_name)
