from typing import Optional, Tuple
import bpy
import math
import numpy as np
from . import xc3_model_py
from mathutils import Matrix
import bmesh


def export_skeleton(armature: bpy.types.Object):
    bpy.ops.object.mode_set(mode="EDIT")

    bones = []
    for bone in armature.data.edit_bones:
        name = bone.name
        if bone.parent:
            matrix = get_bone_transform(bone.parent.matrix.inverted() @ bone.matrix)
            transform = np.array(matrix.transposed())
        else:
            matrix = get_root_bone_transform(bone)
            transform = np.array(matrix.transposed())

        # TODO: Find a way to make this not O(N^2)?
        parent_index = None
        if bone.parent:
            for i, other in enumerate(armature.data.edit_bones):
                if other == bone.parent:
                    parent_index = i
                    break
        bones.append(xc3_model_py.Bone(name, transform, parent_index))

    bpy.ops.object.mode_set(mode="OBJECT")

    return xc3_model_py.Skeleton(bones)


def get_root_bone_transform(bone: bpy.types.EditBone) -> Matrix:
    bone.transform(Matrix.Rotation(math.radians(-90), 4, "X"))
    bone.transform(Matrix.Rotation(math.radians(90), 4, "Z"))
    unreoriented_matrix = get_bone_transform(bone.matrix)
    bone.transform(Matrix.Rotation(math.radians(-90), 4, "Z"))
    bone.transform(Matrix.Rotation(math.radians(90), 4, "X"))
    return unreoriented_matrix


def get_bone_transform(m: Matrix) -> Matrix:
    # This is the inverse of the get_blender_transform permutation matrix.
    # https://en.wikipedia.org/wiki/Matrix_similarity
    p = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Perform the transformation m in Blender's basis and convert back to Ultimate.
    return (p @ m @ p.inverted()).transposed()


def extract_index(name: str) -> Tuple[Optional[int], str]:
    name_parts = name.split(".", 1)

    prefix = None
    name = name_parts[1] if len(name_parts) == 2 else ""

    try:
        prefix = int(name_parts[0])
    except:
        prefix = None

    return prefix, name


# Updated from the processing code written for Smash Ultimate:
# https://github.com/ssbucarlos/smash-ultimate-blender/blob/a003be92bd27e34d2a6377bb98d55d5a34e63e56/source/model/export_model.py#L956
def process_export_mesh(context: bpy.types.Context, mesh: bpy.types.Object):
    # Apply any transforms before exporting to preserve vertex positions.
    # Assume the meshes have no children that would inherit their transforms.
    mesh.data.transform(mesh.matrix_basis)
    mesh.matrix_basis.identity()

    # Apply Modifiers
    override = context.copy()
    override["object"] = mesh
    override["active_object"] = mesh
    override["selected_objects"] = [mesh]
    with context.temp_override(**override):
        for modifier in mesh.modifiers:
            if modifier.type != "ARMATURE":
                bpy.ops.object.modifier_apply(modifier=modifier.name)

    # Get the custom normals from the original mesh.
    # We use the copy here since applying transforms alters the normals.
    loop_normals = np.zeros(len(mesh.data.loops) * 3, dtype=np.float32)
    mesh.data.loops.foreach_get("normal", loop_normals)

    # Transfer the original normals to a custom attribute.
    # This allows us to edit the mesh without affecting custom normals.
    normals_color = mesh.data.attributes.new(
        name="_custom_normals", type="FLOAT_VECTOR", domain="CORNER"
    )
    normals_color.data.foreach_set("vector", loop_normals)

    # Check if any faces are not triangles, and convert them into triangles.
    # TODO: Investigate why this causes issues in game.
    if any(len(f.vertices) != 3 for f in mesh.data.polygons):
        bm = bmesh.new()
        bm.from_mesh(mesh.data)

        bmesh.ops.triangulate(bm, faces=bm.faces[:])

        bm.to_mesh(mesh.data)
        bm.free()

    # Blender stores normals and UVs per loop rather than per vertex.
    # Edges with more than one value per vertex need to be split.
    split_duplicate_loop_attributes(mesh)
    # Rarely this will create some loose verts
    bm = bmesh.new()
    bm.from_mesh(mesh.data)

    unlinked_verts = [v for v in bm.verts if len(v.link_faces) == 0]
    bmesh.ops.delete(bm, geom=unlinked_verts, context="VERTS")

    bm.to_mesh(mesh.data)
    mesh.data.update()
    bm.clear()

    # Extract the custom normals preserved in the custom attribute.
    # Attributes should not be affected by splitting or triangulating.
    # This avoids the datatransfer modifier not handling vertices at the same position.
    loop_normals = np.zeros(len(mesh.data.loops) * 3, dtype=np.float32)
    normals = mesh.data.attributes["_custom_normals"]
    normals.data.foreach_get("vector", loop_normals)

    # Assign the preserved custom normals to the temp mesh.
    mesh.data.normals_split_custom_set(loop_normals.reshape((-1, 3)))
    mesh.data.update()


def split_duplicate_loop_attributes(mesh: bpy.types.Object):
    bm = bmesh.new()
    bm.from_mesh(mesh.data)

    edges_to_split: list[bmesh.types.BMEdge] = []

    add_duplicate_normal_edges(edges_to_split, bm)

    for layer_name in bm.loops.layers.uv.keys():
        uv_layer = bm.loops.layers.uv.get(layer_name)
        add_duplicate_uv_edges(edges_to_split, bm, uv_layer)

    # Duplicate edges cause problems with split_edges.
    edges_to_split = list(set(edges_to_split))

    # Don't modify the mesh if no edges need to be split.
    # This check also seems to prevent a potential crash.
    if len(edges_to_split) > 0:
        bmesh.ops.split_edges(bm, edges=edges_to_split)
        bm.to_mesh(mesh.data)
        mesh.data.update()

    bm.clear()

    # Check if any edges were split.
    return len(edges_to_split) > 0


def add_duplicate_normal_edges(edges_to_split, bm):
    # The original normals are preserved in a custom attribute.
    normal_layer = bm.loops.layers.float_vector.get("_custom_normals")

    # Find edges connected to vertices with more than one normal.
    # This allows converting to per vertex later by splitting edges.
    index_to_normal = {}
    for face in bm.faces:
        for loop in face.loops:
            vertex_index = loop.vert.index
            normal = loop[normal_layer]
            # Small fluctuations in normal vectors are expected during processing.
            # Check if the angle between normals is sufficiently large.
            # Assume normal vectors are normalized to have length 1.0.
            if vertex_index not in index_to_normal:
                index_to_normal[vertex_index] = normal
            elif not math.isclose(
                normal.dot(index_to_normal[vertex_index]),
                1.0,
                abs_tol=0.001,
                rel_tol=0.001,
            ):
                # Get any edges containing this vertex.
                edges_to_split.extend(loop.vert.link_edges)


def add_duplicate_uv_edges(edges_to_split, bm, uv_layer):
    # Blender stores uvs per loop rather than per vertex.
    # Find edges connected to vertices with more than one uv coord.
    # This allows converting to per vertex later by splitting edges.
    index_to_uv = {}
    for face in bm.faces:
        for loop in face.loops:
            vertex_index = loop.vert.index
            uv = loop[uv_layer].uv
            # Use strict equality since UVs are unlikely to change unintentionally.
            if vertex_index not in index_to_uv:
                index_to_uv[vertex_index] = uv
            elif uv != index_to_uv[vertex_index]:
                edges_to_split.extend(loop.vert.link_edges)


def export_mesh(
    context: bpy.types.Context,
    root: xc3_model_py.ModelRoot,
    blender_mesh: bpy.types.Object,
    combined_weights: xc3_model_py.skinning.SkinWeights,
    original_meshes,
    morph_names: list[str],
    create_speff_meshes: bool,
):
    # Work on a copy in case we need to make any changes.
    mesh_copy = blender_mesh.copy()
    mesh_copy.data = blender_mesh.data.copy()

    process_export_mesh(context, mesh_copy)

    mesh_data = blender_mesh.data

    # TODO: Is there a better way to account for the change of coordinates?
    axis_correction = np.array(Matrix.Rotation(math.radians(90), 3, "X"))

    positions = np.zeros(len(mesh_data.vertices) * 3)
    mesh_data.vertices.foreach_get("co", positions)
    positions = positions.reshape((-1, 3)) @ axis_correction

    vertex_indices = np.zeros(len(mesh_data.loops), dtype=np.uint32)
    mesh_data.loops.foreach_get("vertex_index", vertex_indices)

    # Normals are stored per loop instead of per vertex.
    loop_normals = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    mesh_data.loops.foreach_get("normal", loop_normals)
    loop_normals = loop_normals.reshape((-1, 3))

    normals = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    normals[:, :3][vertex_indices] = loop_normals @ axis_correction

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
    tangents[:, :3][vertex_indices] = loop_tangents.reshape((-1, 3)) @ axis_correction
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
    material_index, material_name = extract_index(mesh_data.materials[0].name)
    if material_index is None:
        for i, material in enumerate(root.models.materials):
            if material.name == material_name:
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
    original_mesh_index, _ = extract_index(blender_mesh.name)
    if original_mesh_index is None:
        for i, mesh in original_meshes:
            if mesh.material_index == material_index:
                original_mesh_index = i
                break

    if original_mesh_index is not None:
        original_mesh = original_meshes[original_mesh_index]

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

    mesh_index = len(root.models.models[0].meshes)
    root.models.models[0].meshes.append(mesh)

    # TODO: report a warning if this fails.
    # Materials can share names, so check the base mesh index instead.
    if create_speff_meshes and original_mesh_index is not None:
        # Create speff meshes based on the base mesh index.
        # Existing speff meshes aren't referenced by base_mesh_index and will be ignored.
        for mesh in original_meshes:
            if mesh.base_mesh_index == original_mesh_index:
                speff_mesh = xc3_model_py.Mesh(
                    vertex_buffer_index,
                    index_buffer_index,
                    unk_mesh_index1,
                    mesh.material_index,
                    mesh.flags1,
                    mesh.flags2,
                    lod_item_index,
                    ext_mesh_index,
                    base_mesh_index=mesh_index,
                )
                root.models.models[0].meshes.append(speff_mesh)

    vertex_buffer = xc3_model_py.vertex.VertexBuffer(
        attributes, morph_blend_target, morph_targets, None
    )
    root.buffers.vertex_buffers.append(vertex_buffer)

    index_buffer = xc3_model_py.vertex.IndexBuffer(vertex_indices)
    root.buffers.index_buffers.append(index_buffer)


def export_shape_keys(morph_names, mesh_data, positions, vertex_indices):
    # TODO: Is there a better way to account for the change of coordinates?
    axis_correction = np.array(Matrix.Rotation(math.radians(90), 3, "X"))

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
            morph_positions = morph_positions.reshape((-1, 3)) @ axis_correction

            position_deltas = morph_positions - positions
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
