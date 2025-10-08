from typing import Optional, Tuple
import bpy
import math
import numpy as np
from . import xc3_model_py
from mathutils import Matrix
import bmesh


class ExportException(Exception):
    pass


def export_skeleton(armature: bpy.types.Object):
    bpy.ops.object.mode_set(mode="EDIT")

    bones = []
    for bone in armature.data.edit_bones:
        name = bone.name
        if bone.parent:
            matrix = get_bone_transform(bone.parent.matrix.inverted() @ bone.matrix)
        else:
            matrix = get_root_bone_transform(bone)

        # TODO: Find a way to make this not O(N^2)?
        parent_index = None
        if bone.parent:
            for i, other in enumerate(armature.data.edit_bones):
                if other == bone.parent:
                    parent_index = i
                    break

        translation, rotation, scale = matrix.decompose()
        rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
        transform = xc3_model_py.Transform(translation, rotation, scale)
        bones.append(xc3_model_py.Bone(name, transform, parent_index))

    bpy.ops.object.mode_set(mode="OBJECT")

    return xc3_model_py.Skeleton(bones)


# https://github.com/ssbucarlos/smash-ultimate-blender/blob/ba0c4998ca94190bb601857923054224ea5fb468/source/model/export_model.py#L1461
def get_root_bone_transform(bone: bpy.types.EditBone) -> Matrix:
    bone.transform(Matrix.Rotation(math.radians(-90), 4, "X"))
    bone.transform(Matrix.Rotation(math.radians(90), 4, "Z"))
    unreoriented_matrix = get_bone_transform(bone.matrix)
    bone.transform(Matrix.Rotation(math.radians(-90), 4, "Z"))
    bone.transform(Matrix.Rotation(math.radians(90), 4, "X"))
    return unreoriented_matrix


# https://github.com/ssbucarlos/smash-ultimate-blender/blob/ba0c4998ca94190bb601857923054224ea5fb468/source/model/export_model.py#L1482
def get_bone_transform(m: Matrix) -> Matrix:
    # This is the inverse of the get_blender_transform permutation matrix.
    # https://en.wikipedia.org/wiki/Matrix_similarity
    p = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Perform the transformation m in Blender's basis and convert back to Xenoblade.
    return p @ m @ p.inverted()


def parse_int(name: str) -> Optional[int]:
    value = None
    try:
        value = int(name)
    except:
        value = None

    return value


def extract_name_index(name: str) -> Tuple[str, Optional[int]]:
    # Extract name and index from different naming conventions.
    # Use >= to ignore any additional parts like file extension.
    name_parts = name.split(".")
    if len(name_parts) >= 3:
        # model_name.index.name
        return name_parts[2], parse_int(name_parts[1])
    elif len(name_parts) == 2:
        # index.name
        return name_parts[1], parse_int(name_parts[0])
    elif len(name_parts) == 1:
        # name
        return name, None
    else:
        return "", None


# Updated from the processing code written for Smash Ultimate:
# https://github.com/ssbucarlos/smash-ultimate-blender/blob/a003be92bd27e34d2a6377bb98d55d5a34e63e56/source/model/export_model.py#L956
def process_export_mesh(context: bpy.types.Context, mesh: bpy.types.Object):
    # Apply any transforms before exporting to preserve vertex positions.
    # Assume the meshes have no children that would inherit their transforms.
    mesh.data.transform(mesh.matrix_basis)
    mesh.matrix_basis.identity()

    # Apply modifiers other than armature and outlines.
    override = context.copy()
    override["object"] = mesh
    override["active_object"] = mesh
    override["selected_objects"] = [mesh]
    with context.temp_override(**override):
        for modifier in mesh.modifiers:
            if modifier.type != "ARMATURE" and modifier.type != "SOLIDIFY":
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
    operator: bpy.types.Operator,
    root: xc3_model_py.ModelRoot,
    blender_mesh: bpy.types.Object,
    combined_weights: xc3_model_py.skinning.SkinWeights,
    original_meshes,
    original_materials,
    morph_names: list[str],
    create_speff_meshes: bool,
    image_replacements: set,
):
    # Work on a copy in case we need to make any changes.
    mesh_copy = blender_mesh.copy()
    mesh_copy.data = blender_mesh.data.copy()

    try:
        process_export_mesh(context, mesh_copy)

        export_mesh_inner(
            operator,
            root,
            mesh_copy,
            blender_mesh.name,
            combined_weights,
            original_meshes,
            original_materials,
            morph_names,
            create_speff_meshes,
            image_replacements,
        )
    finally:
        bpy.data.meshes.remove(mesh_copy.data)


# TODO: Split this into more functions.
def export_mesh_inner(
    operator: bpy.types.Operator,
    root: xc3_model_py.ModelRoot,
    blender_mesh: bpy.types.Object,
    mesh_name: str,
    combined_weights: xc3_model_py.skinning.SkinWeights,
    original_meshes,
    original_materials,
    morph_names: list[str],
    create_speff_meshes: bool,
    image_replacements: set,
):

    mesh_data: bpy.types.Mesh = blender_mesh.data

    # This needs to be checked after processing in case there are more vertices.
    # TODO: Support 32 bit indices eventually and make this a warning.
    vertex_count = len(mesh_data.vertices)
    if vertex_count > 65535:
        message = f"Mesh {mesh_name} will have {vertex_count} vertices after exporting,"
        message += " which exceeds the per mesh limit of 65535."
        raise ExportException(message)

    z_up_to_y_up = np.array(Matrix.Rotation(math.radians(90), 3, "X"), dtype=np.float32)

    positions = export_positions(mesh_data, z_up_to_y_up)
    vertex_indices = export_vertex_indices(mesh_data)
    normals = export_normals(mesh_data, z_up_to_y_up, vertex_indices)
    tangents = export_tangents(mesh_data, z_up_to_y_up, vertex_indices)

    # Remove the outline vertex group before exporting weights.
    outline_alpha = export_outline_alpha(blender_mesh, positions)

    influences = export_influences(
        operator, blender_mesh, mesh_data, mesh_name, combined_weights.bone_names
    )
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
        attribute = export_uv_layer(
            mesh_name, mesh_data, positions, vertex_indices, uv_layer
        )
        attributes.append(attribute)

    for color_attribute in mesh_data.color_attributes:
        if color_attribute.name not in ["OutlineVertexColor", "VertexNormal"]:
            attribute = export_color_attribute(
                mesh_name, mesh_data, vertex_indices, color_attribute
            )
            attributes.append(attribute)

    morph_targets = export_shape_keys(
        morph_names, mesh_data, positions, normals, tangents
    )

    morph_blend_target = export_morph_blend_target(
        positions, normals, tangents, morph_targets
    )

    # Give each mesh a unique vertex and index buffer for simplicity.
    vertex_buffer_index = len(root.buffers.vertex_buffers)
    index_buffer_index = len(root.buffers.index_buffers)

    material_index, material_name, is_new_material = extract_material_name_info(
        original_materials, mesh_name, mesh_data
    )

    # TODO: why does None not work well in game?
    lod_item_index = 0
    # TODO: What to use for mesh flags?
    flags1 = 16384
    flags2 = 16385
    ext_mesh_index = None
    base_mesh_index = None
    # Use the index buffer as the shadow map index buffer.
    # We don't use the original index since the new buffers are different.
    index_buffer_index2 = index_buffer_index

    original_mesh_index = extract_mesh_index(mesh_name, original_meshes, material_index)

    if original_mesh_index is not None:
        # Preserve original fields for meshes like "0.material"
        # Perform this before potentially adding materials.
        original_mesh = original_meshes[original_mesh_index]

        lod_item_index = original_mesh.lod_item_index
        flags1 = original_mesh.flags1
        flags2 = original_mesh.flags2
        ext_mesh_index = original_mesh.ext_mesh_index
        base_mesh_index = original_mesh.base_mesh_index

    original_material = original_materials[material_index]
    material_texture_indices = get_texture_assignments(
        mesh_data, original_material, root.image_textures
    )

    # TODO: Share code with editing speff materials.
    if is_new_material:
        # Add a new material with the given name.
        # Avoid potentially referencing an added material here.
        material_to_edit = copy_material(original_material)
        material_to_edit.name = material_name
        material_index = len(root.models.materials)
        root.models.materials.append(material_to_edit)
    else:
        # Update an existing material.
        material_to_edit = root.models.materials[material_index]

    apply_texture_indices(material_to_edit, material_texture_indices)
    apply_toon_gradient_row(mesh_data, material_to_edit)

    for i, image in material_texture_indices.values():
        image_replacements.add((i, image))

    new_mesh = xc3_model_py.Mesh(
        vertex_buffer_index,
        index_buffer_index,
        index_buffer_index2,
        material_index,
        flags1,
        flags2,
        lod_item_index,
        ext_mesh_index,
        base_mesh_index,
    )

    mesh_index = len(root.models.models[0].meshes)
    root.models.models[0].meshes.append(new_mesh)

    # XC1 and XC2 don't use speff meshes.
    only_base_meshes = all(m.base_mesh_index == 0 for m in original_meshes)

    # TODO: report a warning if this fails.
    if create_speff_meshes and not only_base_meshes and original_mesh_index is not None:
        # Materials can share names, so check the base mesh index instead.
        # Existing speff meshes aren't referenced by base_mesh_index and will be ignored.
        # TODO: This can be generated more reliably based on the base material once flags are figured out.
        for mesh in original_meshes:
            if mesh.base_mesh_index == original_mesh_index:
                # Avoid modifying existing speff materials if the material is new.
                speff_material_index = mesh.material_index
                if is_new_material:
                    material_to_edit = copy_material(
                        root.models.materials[mesh.material_index]
                    )
                    speff_material_index = len(root.models.materials)
                    root.models.materials.append(material_to_edit)
                else:
                    material_to_edit = root.models.materials[mesh.material_index]

                apply_texture_indices(material_to_edit, material_texture_indices)
                apply_toon_gradient_row(mesh_data, material_to_edit)

                speff_mesh = copy_mesh(new_mesh)
                speff_mesh.material_index = speff_material_index
                speff_mesh.flags1 = mesh.flags1
                speff_mesh.flags2 = mesh.flags2
                speff_mesh.base_mesh_index = mesh_index
                root.models.models[0].meshes.append(speff_mesh)

    outline_buffer_index = None

    has_outlines = mesh_has_outlines(blender_mesh)

    if has_outlines and original_mesh_index is not None:
        outline_mesh = export_outline_mesh(
            original_meshes,
            original_materials,
            original_mesh_index,
            new_mesh,
            mesh_index,
        )

        if outline_mesh is not None:
            outline_buffer_index = export_outline_buffer(
                root,
                mesh_name,
                mesh_data,
                vertex_indices,
                normals,
                outline_alpha,
                morph_targets,
            )

            root.models.models[0].meshes.append(outline_mesh)
        else:
            message = f"Unable to find outline mesh for {mesh_name} in original wimdo to generate outlines"
            operator.report({"WARNING"}, message)

    vertex_buffer = xc3_model_py.vertex.VertexBuffer(
        attributes, morph_blend_target, morph_targets, outline_buffer_index
    )
    primitive_type = xc3_model_py.vertex.PrimitiveType.TriangleList
    index_buffer = xc3_model_py.vertex.IndexBuffer(
        vertex_indices.astype(np.uint16), primitive_type
    )
    root.buffers.vertex_buffers.append(vertex_buffer)

    root.buffers.index_buffers.append(index_buffer)


def export_outline_mesh(
    original_meshes,
    original_materials,
    original_mesh_index,
    new_mesh,
    mesh_index,
):
    original_mesh = original_meshes[original_mesh_index]

    # Find the original outline mesh and its outline material.
    # TODO: Find a way to generate outline meshes and materials instead.
    original_outline_mesh = None
    outline_material_index = None
    for mesh in original_meshes:
        # Outlines use the same vertex data but a different material.
        if (
            mesh.vertex_buffer_index == original_mesh.vertex_buffer_index
            and mesh.index_buffer_index == original_mesh.index_buffer_index
            and original_materials[mesh.material_index].name.endswith("_outline")
        ):
            original_outline_mesh = mesh
            outline_material_index = mesh.material_index
            break

    if original_outline_mesh is not None and outline_material_index is not None:
        outline_mesh = copy_mesh(new_mesh)
        outline_mesh.material_index = outline_material_index
        outline_mesh.flags1 = original_outline_mesh.flags1
        outline_mesh.flags2 = original_outline_mesh.flags2
        outline_mesh.base_mesh_index = mesh_index
        return outline_mesh
    else:
        return None


def mesh_has_outlines(blender_mesh):
    # TODO: Is there a more reliable way to check for outlines?
    for modifier in blender_mesh.modifiers:
        if modifier.type == "SOLIDIFY":
            return True

    return False


def export_outline_alpha(blender_mesh, positions):
    outline_alpha = None

    # Avoid messing up vertex weights by removing outline information.
    # This is safe since we're working on a copy of the original mesh.
    outline_vertex_group = blender_mesh.vertex_groups.get("OutlineThickness")
    if outline_vertex_group is not None:
        # TODO: Is there a way to do this without looping?
        outline_alpha = np.zeros(positions.shape[0], dtype=np.float32)
        for i in range(positions.shape[0]):
            outline_alpha[i] = outline_vertex_group.weight(i)

        blender_mesh.vertex_groups.remove(outline_vertex_group)

    return outline_alpha


def export_morph_blend_target(positions, normals, tangents, morph_targets):
    morph_blend_target = []
    if len(morph_targets) > 0:
        # TODO: Handle creating attributes and ordering in xc3_model?
        morph_blend_target = [
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Position2, positions
            ),
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Normal4, normals
            ),
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.OldPosition, positions
            ),
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Tangent2, tangents
            ),
        ]

    return morph_blend_target


def export_outline_buffer(
    root, mesh_name, mesh_data, vertex_indices, normals, outline_alpha, morph_targets
):
    # xc3_model will fill in missing required attributes with default values.
    # TODO: Raise an error if the color data is missing?
    # TODO: How to handle the alpha for outline width?
    outline_attributes = []

    # Buffers with morphs should omit the normals to work properly in game.
    # The normals are already provided by the morph attributes.
    if len(morph_targets) == 0:
        outline_attributes.append(
            xc3_model_py.vertex.AttributeData(
                xc3_model_py.vertex.AttributeType.Normal, normals
            )
        )

    for color_attribute in mesh_data.color_attributes:
        if color_attribute.name == "OutlineVertexColor":
            attribute = export_color_attribute(
                mesh_name, mesh_data, vertex_indices, color_attribute
            )
            # Get the outline thickness used by the solidify modifier.
            # Use imported vertex color alpha as a default.
            if outline_alpha is not None:
                attribute.data[:, 3] = outline_alpha

            outline_attributes.append(attribute)

    outline_buffer_index = len(root.buffers.outline_buffers)
    outline_buffer = xc3_model_py.vertex.OutlineBuffer(outline_attributes)
    root.buffers.outline_buffers.append(outline_buffer)

    return outline_buffer_index


def apply_toon_gradient_row(mesh_data, material):
    # Try and find the non processed toon gradient value.
    # This works since type 26 only seems to be used for toon gradients.
    toon_row_index = extract_toon_gradient_row(mesh_data)
    if toon_row_index is not None:
        for c in material.work_callbacks:
            if c.unk1 == 26:
                material.work_values[c.unk2] = toon_row_index
                material.work_values[c.unk2 + 1] = toon_row_index


def extract_mesh_index(mesh_name, original_meshes, material_index):
    _, mesh_index = extract_name_index(mesh_name)
    if mesh_index is None:
        for i, mesh in enumerate(original_meshes):
            if mesh.material_index == material_index:
                mesh_index = i
                break

    return mesh_index


def extract_toon_gradient_row(mesh_data) -> Optional[float]:
    for node in mesh_data.materials[0].node_tree.nodes:
        if node.label == "Toon Gradient Row":
            return node.outputs[0].default_value
    return None


def extract_material_name_info(materials, mesh_name, mesh_data):
    # Use names as a less accurate fallback for the original material.
    blender_material_name = mesh_data.materials[0].name
    material_name, material_index = extract_name_index(blender_material_name)
    is_new_material = True
    for i, material in enumerate(materials):
        if material.name == material_name:
            # TODO: handle setting an existing name to reference a different original material?
            is_new_material = False
            if material_index is None:
                material_index = i
            break

    if material_index is None:
        message = f"Failed to find original material for mesh {mesh_name} with material {blender_material_name}."
        raise ExportException(message)

    if material_index < 0 or material_index >= len(materials):
        message = f"Material index {material_index} for mesh {mesh_name}"
        message += f" does not reference one of {len(materials)} original materials."
        raise ExportException(message)

    return material_index, material_name, is_new_material


def export_influences(
    operator, blender_mesh, mesh_data, mesh_name: str, bone_names: list[str]
):
    # Export Weights
    # TODO: Reversing a vertex -> group lookup to a group -> vertex lookup is expensive.
    # TODO: Does Blender not expose this directly?
    group_to_weights = {
        vg.index: (vg.name, [])
        for vg in blender_mesh.vertex_groups
        if vg.name != "OutlineThickness"
    }

    for vertex in mesh_data.vertices:
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
        if name not in bone_names:
            message = f"Vertex group {name} for {mesh_name} is not in original wimdo bone list and will be skipped."
            operator.report({"WARNING"}, message)
        elif len(weights) > 0:
            influence = xc3_model_py.skinning.Influence(name, weights)
            influences.append(influence)

    return influences


def export_positions(mesh_data, z_up_to_y_up):
    positions = np.zeros(len(mesh_data.vertices) * 3, dtype=np.float32)
    mesh_data.vertices.foreach_get("co", positions)
    positions = positions.reshape((-1, 3)) @ z_up_to_y_up
    return positions


def export_vertex_indices(mesh_data):
    vertex_indices = np.zeros(len(mesh_data.loops), dtype=np.uint32)
    mesh_data.loops.foreach_get("vertex_index", vertex_indices)
    return vertex_indices


def export_normals(mesh_data, z_up_to_y_up, vertex_indices):
    # Normals are stored per loop instead of per vertex.
    loop_normals = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    mesh_data.loops.foreach_get("normal", loop_normals)
    loop_normals = loop_normals.reshape((-1, 3))

    normals = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    normals[:, :3][vertex_indices] = loop_normals @ z_up_to_y_up

    # Some shaders use the 4th component for normal map intensity.
    if attribute := mesh_data.attributes.get("VertexNormal"):
        vertex_normals = export_per_vertex_colors(mesh_data, vertex_indices, attribute)
        normals[:, 3] = vertex_normals[:, 3]
    else:
        normals[:, 3] = 1.0

    return normals


def export_tangents(mesh_data, z_up_to_y_up, vertex_indices):
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
    tangents[:, :3][vertex_indices] = loop_tangents.reshape((-1, 3)) @ z_up_to_y_up
    tangents[:, 3][vertex_indices] = loop_bitangent_signs
    return tangents


def export_color_attribute(mesh_name, mesh_data, vertex_indices, color_attribute):
    ty = xc3_model_py.vertex.AttributeType.VertexColor
    match color_attribute.name:
        case "VertexColor":
            ty = xc3_model_py.vertex.AttributeType.VertexColor
        case "Blend":
            ty = xc3_model_py.vertex.AttributeType.Blend
        case "OutlineVertexColor":
            ty = xc3_model_py.vertex.AttributeType.VertexColor
        case _:
            message = f'"{color_attribute.name}" for mesh {mesh_name} is not one of the supported color attribute names.'
            message += ' Valid names are "VertexColor" and "Blend".'
            raise ExportException(message)

    colors = export_per_vertex_colors(mesh_data, vertex_indices, color_attribute)
    a = xc3_model_py.vertex.AttributeData(ty, colors)
    return a


def export_per_vertex_colors(mesh_data, vertex_indices, color_attribute):
    # TODO: error for unsupported data_type.
    if color_attribute.domain == "POINT":
        colors = np.zeros(len(mesh_data.vertices) * 4, dtype=np.float32)
        color_attribute.data.foreach_get("color", colors)
    elif color_attribute.domain == "CORNER":
        loop_colors = np.zeros(len(mesh_data.loops) * 4, dtype=np.float32)
        color_attribute.data.foreach_get("color", loop_colors)
        # Convert per loop data to per vertex data.
        colors = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
        colors[vertex_indices] = loop_colors.reshape((-1, 4))
    else:
        message = f"Unsupported color attribute domain {color_attribute.domain}"
        raise ExportException(message)

    return colors.reshape((-1, 4))


def export_uv_layer(mesh_name, mesh_data, positions, vertex_indices, uv_layer):
    texcoords = np.zeros((positions.shape[0], 2), dtype=np.float32)
    loop_uvs = np.zeros(len(mesh_data.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", loop_uvs)
    texcoords[vertex_indices] = loop_uvs.reshape((-1, 2))
    # Flip vertically to match in game.
    texcoords[:, 1] = 1.0 - texcoords[:, 1]

    ty = xc3_model_py.vertex.AttributeType.TexCoord0
    match uv_layer.name:
        case "TexCoord0":
            ty = xc3_model_py.vertex.AttributeType.TexCoord0
        case "TexCoord1":
            ty = xc3_model_py.vertex.AttributeType.TexCoord1
        case "TexCoord2":
            ty = xc3_model_py.vertex.AttributeType.TexCoord2
        case "TexCoord3":
            ty = xc3_model_py.vertex.AttributeType.TexCoord3
        case "TexCoord4":
            ty = xc3_model_py.vertex.AttributeType.TexCoord4
        case "TexCoord5":
            ty = xc3_model_py.vertex.AttributeType.TexCoord5
        case "TexCoord6":
            ty = xc3_model_py.vertex.AttributeType.TexCoord6
        case "TexCoord7":
            ty = xc3_model_py.vertex.AttributeType.TexCoord7
        case "TexCoord8":
            ty = xc3_model_py.vertex.AttributeType.TexCoord8
        case _:
            message = f'"{uv_layer.name}" for mesh {mesh_name} is not one of the supported UV map names.'
            message += ' Valid names are "TexCoord0" to "TexCoord8".'
            raise ExportException(message)

    return xc3_model_py.vertex.AttributeData(ty, texcoords)


def apply_texture_indices(material, indices):
    for texture in material.textures:
        image_index_image = indices.get(texture.image_texture_index)
        if image_index_image is not None:
            image_index, _ = image_index_image
            texture.image_texture_index = image_index


# TODO: also get the images themselves
def get_texture_assignments(mesh_data, material, image_textures):
    old_to_new_index = {}
    # TODO: error if there are no nodes or not enough textures?
    for node in mesh_data.materials[0].node_tree.nodes:
        if node.bl_idname != "ShaderNodeTexImage":
            continue

        # Update material texture assignments.
        # Support the old labels like "0" or new labels like "s0".
        label = node.label.lstrip("s")
        texture_index = parse_int(label)
        if texture_index is None:
            continue

        image_index = image_index_to_replace(image_textures, node.image.name)

        if image_index is not None:
            try:
                index = material.textures[texture_index].image_texture_index
                old_to_new_index[index] = (
                    image_index,
                    node.image,
                )
            except:
                # TODO: how to handle combining materials from two models?
                pass

    return old_to_new_index


def image_index_to_replace(images, image_name: str) -> Optional[int]:
    # Find the original image to replace.
    # TODO: handle new images without an index?
    image_name, image_index = extract_name_index(image_name)
    if image_index is None:
        for i, image in enumerate(images):
            if image.name == image_name:
                image_index = i
                break

    return image_index


def export_shape_keys(
    morph_names: list[str],
    mesh_data,
    positions: np.ndarray,
    normals: np.ndarray,
    tangents: np.ndarray,
):
    z_up_to_y_up = np.array(Matrix.Rotation(math.radians(90), 3, "X"), dtype=np.float32)

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

            morph_positions = np.zeros(len(mesh_data.vertices) * 3, dtype=np.float32)
            shape_key.points.foreach_get("co", morph_positions)
            morph_positions = morph_positions.reshape((-1, 3)) @ z_up_to_y_up

            position_deltas = morph_positions - positions

            # TODO: Calculate better values for normals and tangents?

            # Calculate sparse indices to avoid including zero elements.
            # TODO: Is there a better threshold value?
            vertex_indices = np.where(
                ~np.all(np.isclose(position_deltas, 0, atol=1e-6), axis=1)
            )[0]

            target = xc3_model_py.vertex.MorphTarget(
                morph_controller_index,
                position_deltas[vertex_indices],
                normals[vertex_indices],
                tangents[vertex_indices],
                vertex_indices.astype(np.uint32),
            )
            morph_targets.append(target)

    return morph_targets


def copy_material(material: xc3_model_py.material.Material):
    # TODO: does pyo3 support deep copy?
    # TODO: Add deep copy support to xc3_model_py?
    textures = [
        xc3_model_py.material.Texture(t.image_texture_index, t.sampler_index)
        for t in material.textures
    ]
    return xc3_model_py.material.Material(
        material.name,
        material.flags,
        material.render_flags,
        material.state_flags,
        material.color,
        textures,
        np.array(material.work_values, dtype=np.float32),
        material.shader_vars,
        material.work_callbacks,
        material.alpha_test_ref,
        material.m_unks1_1,
        material.m_unks1_2,
        material.m_unks1_3,
        material.m_unks1_4,
        material.technique_index,
        material.pass_type,
        material.parameters,
        material.m_unks2_2,
        material.gbuffer_flags,
        material.alpha_test,
        material.shader,
        material.fur_params,
    )


def copy_mesh(mesh: xc3_model_py.Mesh) -> xc3_model_py.Mesh:
    return xc3_model_py.Mesh(
        mesh.vertex_buffer_index,
        mesh.index_buffer_index,
        mesh.index_buffer_index2,
        mesh.material_index,
        mesh.flags1,
        mesh.flags2,
        mesh.lod_item_index,
        mesh.ext_mesh_index,
        mesh.base_mesh_index,
    )
