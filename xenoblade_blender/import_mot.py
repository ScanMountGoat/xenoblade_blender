import bpy
import time
import logging
import numpy as np
import math

from . import xc3_model_py
from .export_root import export_skeleton

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from mathutils import Matrix


class ImportMot(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade animation"""

    bl_idname = "import_scene.mot"
    bl_label = "Import Mot"

    filename_ext = ".mot"

    filter_glob: StringProperty(
        default="*.mot",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=log_fmt, level=logging.INFO)

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

    # TODO: Will this order always match the xc3_model skeleton order?
    # TODO: Will bones always appear after their parents?
    bone_names = [bone.name for bone in skeleton.bones]
    for i, bone in enumerate(skeleton.bones):
        if bone.parent_index is not None and bone.parent_index > i:
            print(f"invalid index {bone.parent_index} > {i}")
    hash_to_name = {xc3_model_py.animation.murmur3(name): name for name in bone_names}

    # TODO: Is this the best way to load all animations?
    # TODO: Optimize this.
    for animation in animations:
        action = import_animation(
            armature, skeleton, bone_names, hash_to_name, animation
        )
        armature.animation_data.action = action

    end = time.time()
    print(f"Import Blender Animation: {end - start}")


def import_animation(armature, skeleton, bone_names, hash_to_name, animation):
    action = bpy.data.actions.new(animation.name)
    if animation.frame_count > 0:
        action.frame_end = float(animation.frame_count) - 1.0

    # Assume each bone appears in only one track.
    animated_bones = {
        get_bone_name(t, bone_names, hash_to_name) for t in animation.tracks
    }

    # Collect keyframes for the appropriate bones.
    positions = {name: [] for name in animated_bones}
    rotations_wxyz = {name: [] for name in animated_bones}
    scales = {name: [] for name in animated_bones}

    for frame in range(animation.frame_count):
        # Use xc3_model_py to efficiently handle different anim types.
        # Transforms need to be relative to the parent bone to animate properly.
        transforms = animation.local_space_transforms(skeleton, frame)

        for name, transform in zip(bone_names, transforms):
            if name not in animated_bones:
                continue

            pose_bone = armature.pose.bones.get(name)
            matrix = Matrix(transform).transposed()
            if pose_bone.parent is not None:
                pose_bone.matrix = pose_bone.parent.matrix @ get_blender_transform(
                    matrix
                )
            else:
                y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")
                x_major_to_y_major = Matrix.Rotation(math.radians(-90), 4, "Z")
                pose_bone.matrix = y_up_to_z_up @ matrix @ x_major_to_y_major

            t, r, s = pose_bone.matrix_basis.decompose()
            positions[name].append(t)
            rotations_wxyz[name].append(r)
            scales[name].append(s)

    for name in animated_bones:
        if name is None:
            continue

        # TODO: Use actual cubic keyframes instead of baking at each frame?
        set_fcurves(action, name, "location", positions[name], 3)
        set_fcurves(action, name, "rotation_quaternion", rotations_wxyz[name], 4)
        set_fcurves(action, name, "scale", scales[name], 3)
    return action


def get_blender_transform(m):
    # In game, the bone's x-axis points from parent to child.
    # In Blender, the bone's y-axis points from parent to child.
    # https://en.wikipedia.org/wiki/Matrix_similarity
    p = Matrix([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Perform the transformation m in Ultimate's basis and convert back to Blender.
    return p @ m @ p.inverted()


def get_bone_name(track, bone_names: list[str], hash_to_name):
    bone_index = track.bone_index()
    bone_hash = track.bone_hash()
    bone_name = track.bone_name()
    if bone_index is not None:
        # TODO: Does the armature preserve insertion order?
        return bone_names[bone_index]
    elif bone_hash is not None:
        if bone_hash in hash_to_name:
            return hash_to_name[bone_hash]
    elif bone_name is not None:
        return bone_name

    return None


def set_fcurves(action, bone_name: str, value_name: str, values, component_count):
    for i in range(component_count):
        # Values can be quickly set in the form [frame, value, frame, value, ...]
        # Assume one value at each frame index for now.
        keyframe_points = [
            val for pair in enumerate(v[i] for v in values) for val in pair
        ]

        # Each coordinate of each value has its own fcurve.
        data_path = f'pose.bones["{bone_name}"].{value_name}'
        fcurve = action.fcurves.new(data_path, index=i, action_group=bone_name)
        fcurve.keyframe_points.add(count=len(values))
        fcurve.keyframe_points.foreach_set("co", keyframe_points)
