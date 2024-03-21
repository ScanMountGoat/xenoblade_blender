import numpy as np
from . import xc3_model_py


def export_skeleton(armature):
    bones = []
    for bone in armature.data.bones.values():
        name = bone.name
        transform = np.array(bone.matrix_local.transposed())
        if bone.parent:
            matrix = bone.parent.matrix_local.inverted() @ bone.matrix_local
            transform = np.array(matrix.transposed())

        # TODO: Find a way to make this not O(N^2)?
        parent_index = None
        if bone.parent:
            for i, other in enumerate(armature.data.bones.values()):
                if other == bone.parent:
                    parent_index = i
                    break
        bones.append(xc3_model_py.Bone(name, transform, parent_index))

    return xc3_model_py.Skeleton(bones)
