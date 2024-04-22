import bpy
import math
import numpy as np
from . import xc3_model_py
from mathutils import Matrix


def export_skeleton(armature: bpy.types.Object):
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


def export_mesh(root: xc3_model_py.ModelRoot, blender_mesh: bpy.types.Object):
    mesh_data = blender_mesh.data

    # TODO: Is there a better way to account for the change of coordinates?
    axis_correction = np.array(Matrix.Rotation(math.radians(90), 3, 'X'))

    positions = np.zeros(len(mesh_data.vertices) * 3)
    mesh_data.vertices.foreach_get('co', positions)
    positions = positions.reshape((-1, 3)) @ axis_correction

    vertex_indices = np.zeros(len(mesh_data.loops), dtype=np.uint32)
    mesh_data.loops.foreach_get('vertex_index', vertex_indices)

    # Normals are stored per loop instead of per vertex.
    loop_normals = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    mesh_data.loops.foreach_get('normal', loop_normals)
    loop_normals = loop_normals.reshape((-1, 3)) @ axis_correction

    normals = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    normals[:, :3][vertex_indices] = loop_normals

    # Tangents are stored per loop instead of per vertex.
    mesh_data.calc_tangents()
    loop_tangents = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    mesh_data.loops.foreach_get('tangent', loop_tangents)

    loop_bitangent_signs = np.zeros(len(mesh_data.loops), dtype=np.float32)
    mesh_data.loops.foreach_get('bitangent_sign', loop_bitangent_signs)

    tangents = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    tangents[:, :3][vertex_indices] = loop_tangents.reshape(
        (-1, 3)) @ axis_correction
    tangents[:, 3][vertex_indices] = loop_bitangent_signs

    # TODO: multiple UV and color attributes
    texcoords = np.zeros((positions.shape[0], 2), dtype=np.float32)
    uv_layer = mesh_data.uv_layers[0]
    loop_uvs = np.zeros(len(mesh_data.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", loop_uvs)
    texcoords[vertex_indices] = loop_uvs.reshape((-1, 2))
    # Flip vertically to match in game.
    texcoords[:, 1] = 1.0 - texcoords[:, 1]

    # TODO: create influences and then convert to weights
    # TODO: Where to store weights for each mesh?
    # TODO: Don't assume only one bone.
    weight_indices = np.zeros((positions.shape[0], 2), dtype=np.uint16)

    # TODO: Does this order need to match in game to work properly?
    attributes = [
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.Position, positions),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.WeightIndex, weight_indices),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.TexCoord0, texcoords),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.Normal, normals),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.Tangent, tangents),
    ]

    # Give each mesh a unique vertex and index buffer for simplicity.
    vertex_buffer_index = len(root.groups[0].buffers[0].vertex_buffers)
    index_buffer_index = len(root.groups[0].buffers[0].index_buffers)
    # Don't support adding new materials for now.
    # xc3_model doesn't actually overwrite materials yet.
    material_index = 0
    for i, material in enumerate(root.groups[0].models[0].materials):
        if material.name == mesh_data.materials[0].name:
            material_index = i
            break

    # TODO: What to use for mesh flags?
    lod = 1
    flags1 = 24576
    flags2 = 16400
    mesh = xc3_model_py.Mesh(vertex_buffer_index, index_buffer_index, material_index, lod, flags1, flags2)

    vertex_buffer = xc3_model_py.vertex.VertexBuffer(attributes, [], None)
    index_buffer = xc3_model_py.vertex.IndexBuffer(vertex_indices)

    root.groups[0].buffers[0].vertex_buffers.append(vertex_buffer)
    root.groups[0].buffers[0].index_buffers.append(index_buffer)
    root.groups[0].models[0].models[0].meshes.append(mesh)
