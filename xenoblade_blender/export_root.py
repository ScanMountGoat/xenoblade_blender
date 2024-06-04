from typing import Optional
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


def extract_index(name: str) -> Optional[int]:
    name_parts = name.split(".")
    try:
        return int(name_parts[0])
    except:
        return None


def export_mesh(
    root: xc3_model_py.ModelRoot,
    blender_mesh: bpy.types.Object,
    combined_weights: xc3_model_py.skinning.SkinWeights,
    original_meshes,
    morph_names: list[str],
):
    mesh_data = blender_mesh.data

    positions = np.zeros(len(mesh_data.vertices) * 3)
    mesh_data.vertices.foreach_get("co", positions)
    positions = positions.reshape((-1, 3))

    vertex_indices = np.zeros(len(mesh_data.loops), dtype=np.uint32)
    mesh_data.loops.foreach_get("vertex_index", vertex_indices)

    # Normals are stored per loop instead of per vertex.
    loop_normals = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    mesh_data.loops.foreach_get("normal", loop_normals)
    loop_normals = loop_normals.reshape((-1, 3))

    normals = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    normals[:, :3][vertex_indices] = loop_normals

    # Tangents are stored per loop instead of per vertex.
    loop_tangents = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    try:
        # TODO: Why do some meshes not have UVs for Pyra?
        mesh_data.calc_tangents()
        mesh_data.loops.foreach_get("tangent", loop_tangents)
    except:
        pass

    loop_bitangent_signs = np.zeros(len(mesh_data.loops), dtype=np.float32)
    mesh_data.loops.foreach_get("bitangent_sign", loop_bitangent_signs)

    tangents = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    tangents[:, :3][vertex_indices] = loop_tangents.reshape((-1, 3))
    tangents[:, 3][vertex_indices] = loop_bitangent_signs

    # Export Weights
    # TODO: Reversing a vertex -> group lookup to a group -> vertex lookup is expensive.
    # TODO: Does Blender not expose this directly?
    group_to_weights = {vg.index: (vg.name, []) for vg in blender_mesh.vertex_groups}

    for vertex in blender_mesh.data.vertices:
        # Blender doesn't enforce normalization, since it normalizes while animating.
        # Normalize on export to ensure the weights work correctly in game.
        weight_sum = sum([g.weight for g in vertex.groups])
        for group in vertex.groups:
            weight = xc3_model_py.skinning.VertexWeight(
                vertex.index, group.weight / weight_sum
            )
            group_to_weights[group.group][1].append(weight)

    influences = []
    for name, weights in group_to_weights.values():
        if len(weights) > 0:
            influence = xc3_model_py.skinning.Influence(name, weights)
            influences.append(influence)

    weight_indices = combined_weights.add_influences(influences, positions.shape[0])

    # Export all available vertex attributes.
    # xc3_model will handle ordering and selecting attributes required by the shader.
    attributes = [
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.Position, positions
        ),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.WeightIndex, weight_indices
        ),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.Normal, normals
        ),
        xc3_model_py.vertex.AttributeData(
            xc3_model_py.vertex.AttributeType.Tangent, tangents
        ),
    ]

    for uv_layer in mesh_data.uv_layers:
        texcoords = np.zeros((positions.shape[0], 2), dtype=np.float32)
        loop_uvs = np.zeros(len(mesh_data.loops) * 2, dtype=np.float32)
        uv_layer.data.foreach_get("uv", loop_uvs)
        texcoords[vertex_indices] = loop_uvs.reshape((-1, 2))
        # Flip vertically to match in game.
        texcoords[:, 1] = 1.0 - texcoords[:, 1]

        ty = xc3_model_py.vertex.AttributeType.TexCoord0
        if uv_layer.name == "TexCoord0":
            ty = xc3_model_py.vertex.AttributeType.TexCoord0
        elif uv_layer.name == "TexCoord1":
            ty = xc3_model_py.vertex.AttributeType.TexCoord1
        elif uv_layer.name == "TexCoord2":
            ty = xc3_model_py.vertex.AttributeType.TexCoord2
        elif uv_layer.name == "TexCoord3":
            ty = xc3_model_py.vertex.AttributeType.TexCoord3
        elif uv_layer.name == "TexCoord4":
            ty = xc3_model_py.vertex.AttributeType.TexCoord4
        elif uv_layer.name == "TexCoord5":
            ty = xc3_model_py.vertex.AttributeType.TexCoord5
        elif uv_layer.name == "TexCoord6":
            ty = xc3_model_py.vertex.AttributeType.TexCoord6
        elif uv_layer.name == "TexCoord7":
            ty = xc3_model_py.vertex.AttributeType.TexCoord7
        elif uv_layer.name == "TexCoord8":
            ty = xc3_model_py.vertex.AttributeType.TexCoord8

        attributes.append(xc3_model_py.vertex.AttributeData(ty, texcoords))

    for color_attribute in mesh_data.color_attributes:
        ty = xc3_model_py.vertex.AttributeType.VertexColor
        if color_attribute.name == "VertexColor":
            ty = xc3_model_py.vertex.AttributeType.VertexColor
        elif color_attribute.name == "Blend":
            ty = xc3_model_py.vertex.AttributeType.Blend

        # TODO: error for unsupported data_type or domain.
        if color_attribute.domain == "POINT":
            colors = np.zeros(len(mesh_data.vertices) * 4)
            color_attribute.data.foreach_get("color", colors)
        elif color_attribute.domain == "CORNER":
            loop_colors = np.zeros(len(mesh_data.loops) * 4)
            color_attribute.data.foreach_get("color", loop_colors)
            # Convert per loop data to per vertex data.
            colors = np.zeros((len(mesh_data.vertices), 4))
            colors[vertex_indices] = loop_colors.reshape((-1, 4))

        colors = colors.reshape((-1, 4))
        attributes.append(xc3_model_py.vertex.AttributeData(ty, colors))

    morph_targets = export_shape_keys(morph_names, mesh_data, positions, vertex_indices)

    morph_blend_target = []
    if len(morph_targets) > 0:
        # TODO: Handle creating attributes and ordering in xc3_model?
        morph_blend_target = [
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Position2, positions
            ),
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Normal4, normals * 0.5 + 0.5
            ),
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.OldPosition, positions
            ),
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Tangent2, tangents * 0.5 + 0.5
            ),
        ]

    # Give each mesh a unique vertex and index buffer for simplicity.
    vertex_buffer_index = len(root.buffers.vertex_buffers)
    index_buffer_index = len(root.buffers.index_buffers)

    # Don't support adding new materials for now.
    # xc3_model doesn't actually overwrite materials yet.
    material_index = extract_index(mesh_data.materials[0].name)
    if material_index is None:
        for i, material in enumerate(root.models.materials):
            if material.name == mesh_data.materials[0].name:
                material_index = i
                break

    # TODO: why does None not work well in game?
    lod_item_index = 0
    # TODO: What to use for mesh flags?
    flags1 = 16384
    flags2 = 16385
    ext_mesh_index = None
    base_mesh_index = None
    # Use the index buffer as the shadow map index buffer.
    # We don't use the original index since the new buffers are different.
    unk_mesh_index1 = index_buffer_index

    # Preserve original fields for meshes like "0.material"
    mesh_index = extract_index(blender_mesh.name)
    if mesh_index is not None:
        original_mesh = original_meshes[mesh_index]

        lod_item_index = original_mesh.lod_item_index
        flags1 = original_mesh.flags1
        flags2 = original_mesh.flags2
        ext_mesh_index = original_mesh.ext_mesh_index
        base_mesh_index = original_mesh.base_mesh_index

    mesh = xc3_model_py.Mesh(
        vertex_buffer_index,
        index_buffer_index,
        unk_mesh_index1,
        material_index,
        flags1,
        flags2,
        lod_item_index,
        ext_mesh_index,
        base_mesh_index,
    )

    vertex_buffer = xc3_model_py.vertex.VertexBuffer(
        attributes, morph_blend_target, morph_targets, None
    )
    index_buffer = xc3_model_py.vertex.IndexBuffer(vertex_indices)

    root.buffers.vertex_buffers.append(vertex_buffer)
    root.buffers.index_buffers.append(index_buffer)
    root.models.models[0].meshes.append(mesh)


def export_shape_keys(morph_names, mesh_data, positions, vertex_indices):
    morph_targets = []
    if mesh_data.shape_keys is not None:
        for shape_key in mesh_data.shape_keys.key_blocks:
            if shape_key.name == "Basis":
                continue

            # Only add existing morph targets for now.
            morph_controller_index = None
            for i, name in enumerate(morph_names):
                if shape_key.name == name:
                    morph_controller_index = i
                    break

            if morph_controller_index is None:
                continue

            # TODO: make these sparse if vertices are unchanged?
            morph_positions = np.zeros(len(mesh_data.vertices) * 3)
            shape_key.points.foreach_get("co", morph_positions)

            position_deltas = morph_positions.reshape((-1, 3)) - positions
            # TODO: Can these be calculated from Blender?
            normal_deltas = np.zeros((len(mesh_data.vertices), 4))
            tangent_deltas = np.zeros((len(mesh_data.vertices), 4))

            target = xc3_model_py.vertex.MorphTarget(
                morph_controller_index,
                position_deltas,
                normal_deltas,
                tangent_deltas,
                vertex_indices,
            )
            morph_targets.append(target)

    return morph_targets
