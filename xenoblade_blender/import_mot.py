import bpy
import time
import logging

from . import xc3_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from mathutils import Matrix, Quaternion, Vector


class ImportMot(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade animation"""
    bl_idname = "import_scene.mot"
    bl_label = "Import Mot"

    filename_ext = ".mot"

    filter_glob: StringProperty(
        default="*.mot",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        import_mot(context, self.filepath)
        return {'FINISHED'}


def import_mot(context: bpy.types.Context, path: str):
    start = time.time()

    animations = xc3_model_py.load_animations(path)

    end = time.time()
    print(f"Load {len(animations)} Animations: {end - start}")

    start = time.time()

    # TODO: Don't assume an armature is selected?
    armature = context.object
    if armature.animation_data is None:
        armature.animation_data_create()

    # TODO: Is this the best way to load all animations?
    for animation in animations:
        action = bpy.data.actions.new(animation.name)

        # TODO: Will this order always match the xc3_model skeleton order?
        bone_names = [bone.name for bone in armature.data.bones.values()]
        hash_to_name = {xc3_model_py.murmur3(
            name): name for name in bone_names}

        for track in animation.tracks:
            import_track(track, animation, armature,
                         action, bone_names, hash_to_name)

        armature.animation_data.action = action

    end = time.time()
    print(f"Import Blender Animation: {end - start}")


def import_track(track, animation, armature, action, bone_names, hash_to_name):
    # TODO: Handle missing bones?
    bone = get_bone_name(track, bone_names, hash_to_name)
    if bone is None:
        print("Failed to assign track for bone")
        return

    # Assume each bone appears in only one track.
    # TODO: Use actual cubic keyframes instead of baking at each frame?
    pose_bone = armature.pose.bones.get(bone)
    if pose_bone is not None:
        # Workaround for fcurves setting the values for bone.matrix_basis.
        # These values are relative to the parent transform and bone rest pose.
        # TODO: handle this in xc3_model_py instead?
        positions = []
        rotations_wxyz = []
        scales = []

        count = animation.frame_count
        for frame in range(count):
            position = track.sample_translation(frame)
            rotation_xyzw = track.sample_rotation(frame)
            scale = track.sample_scale(frame)

            matrix = animation_transform(position, rotation_xyzw, scale)
            if pose_bone.parent is not None:
                pose_bone.matrix = pose_bone.parent.matrix @ matrix
            else:
                pose_bone.matrix = matrix

            t, r, s = pose_bone.matrix_basis.decompose()
            positions.append(t)
            rotations_wxyz.append(r)
            scales.append(s)

        # Assume each bone appears in only one track.
        # TODO: Use actual cubic keyframes instead of baking at each frame?
        set_fcurves(action, bone, "location", positions, 3)
        set_fcurves(action, bone, "rotation_quaternion", rotations_wxyz, 4)
        set_fcurves(action, bone, "scale", scales, 3)


def animation_transform(translation, rotation_xyzw, scale) -> Matrix:
    tm = Matrix.Translation(translation)
    qr = Quaternion([rotation_xyzw[3], rotation_xyzw[0],
                    rotation_xyzw[1], rotation_xyzw[2]])
    rm = Matrix.Rotation(qr.angle, 4, qr.axis)
    # Blender doesn't have this built in for some reason.
    sm = Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))
    return tm @ rm @ sm


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
        # Each coordinate of each value has its own fcurve.
        data_path = f"pose.bones[\"{bone_name}\"].{value_name}"
        fcurve = action.fcurves.new(data_path, index=i, action_group=bone_name)
        fcurve.keyframe_points.add(count=len(values))
        # TODO: List comprehension?
        # Values can be quickly set in the form [frame, value, frame, value, ...]
        # Assume one value at each frame index for now.
        keyframe_points = []
        for frame, value in enumerate(values):
            keyframe_points.append(frame)
            keyframe_points.append(value[i])
        fcurve.keyframe_points.foreach_set('co', keyframe_points)
