from typing import Optional
import bpy
import numpy as np
import os
import math

from . import xc3_model_py
from mathutils import Matrix
from bpy_extras import image_utils
from pathlib import Path


def get_database_path(version: str) -> str:
    files = {"XC1": "xc1.json", "XC2": "xc2.json", "XC3": "xc3.json"}
    return os.path.join(os.path.dirname(__file__), files[version])


def get_image_folder(image_folder: str, filepath: str) -> str:
    if image_folder == "":
        return str(Path(filepath).parent)
    else:
        return image_folder


# https://github.com/ssbucarlos/smash-ultimate-blender/blob/a003be92bd27e34d2a6377bb98d55d5a34e63e56/source/model/import_model.py#L371
def import_armature(context, root, name: str):
    armature = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.collection.objects.link(armature)

    armature.data.display_type = "STICK"
    armature.rotation_mode = "QUATERNION"
    armature.show_in_front = True

    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT", toggle=False)

    if root.skeleton is not None:
        transforms = root.skeleton.model_space_transforms()

        for bone, transform in zip(root.skeleton.bones, transforms):
            new_bone = armature.data.edit_bones.new(name=bone.name)
            new_bone.head = [0, 0, 0]
            new_bone.tail = [0, 1, 0]
            matrix = Matrix(transform).transposed()
            y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")
            x_major_to_y_major = Matrix.Rotation(math.radians(-90), 4, "Z")
            new_bone.matrix = y_up_to_z_up @ matrix @ x_major_to_y_major

        for bone in root.skeleton.bones:
            if bone.parent_index is not None:
                parent_bone_name = root.skeleton.bones[bone.parent_index].name
                parent_bone = armature.data.edit_bones.get(parent_bone_name)
                armature.data.edit_bones.get(bone.name).parent = parent_bone

        # TODO: Adjust length without causing twisting in animations like bl000101.
        for bone in armature.data.edit_bones:
            # if len(bone.children) > 0:
            #     bone.length = (bone.head - bone.children[0].head).length
            # elif bone.parent:
            #     bone.length = bone.parent.length
            bone.length = 0.1

            # Prevent Blender from removing any bones.
            if bone.length < 0.01:
                bone.length = 0.01

    bpy.ops.object.mode_set(mode="OBJECT")

    return armature


def import_images(root, model_name: str, pack: bool, image_folder: str, flip: bool):
    blender_images = []

    if pack:
        decoded_images = xc3_model_py.decode_images_rgbaf32(root.image_textures)
        for i, (image, decoded) in enumerate(zip(root.image_textures, decoded_images)):
            # Use the same naming conventions as the saved PNG images and xc3_tex.
            if image.name is not None:
                name = f"{model_name}.{i}.{image.name}"
            else:
                name = f"{model_name}.{i}"

            blender_image = bpy.data.images.new(
                name, image.width, image.height, alpha=True
            )

            # TODO: why is this necessary?
            decoded_size = image.width * image.height * 4
            decoded = decoded[:decoded_size]

            if flip:
                # Flip vertically to match Blender.
                decoded = decoded.reshape((image.height, image.width, 4))
                decoded = np.flip(decoded, axis=0)

            blender_image.pixels.foreach_set(decoded.reshape(-1))

            # TODO: This should depend on srgb vs linear in format.
            blender_image.colorspace_settings.is_data = True

            # Pack textures to avoid the prompt to save on exit.
            blender_image.pack()
            blender_images.append(blender_image)
    else:
        # Unpacked textures use less memory and are faster to load.
        # This already includes the index in the file name.
        for image, file in zip(
            root.image_textures,
            root.save_images_rgba8(image_folder, model_name, "png", not flip),
        ):
            blender_image = image_utils.load_image(
                file, place_holder=True, check_existing=False
            )

            # TODO: This should depend on srgb vs linear in format.
            blender_image.colorspace_settings.is_data = True

            blender_images.append(blender_image)

    return blender_images


def import_map_root(
    operator,
    root,
    root_collection: bpy.types.Collection,
    blender_images,
    import_all_meshes: bool,
    flip_uvs: bool,
):
    # Convert from Y up to Z up.
    y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")

    # TODO: Create a group collection?
    for group in root.groups:
        for models in group.models:
            base_lods = None
            if models.lod_data is not None:
                base_lods = [g.base_lod_index for g in models.lod_data.groups]

            # TODO: Cache based on vertex and index buffer indices?
            for model in models.models:
                model_collection = bpy.data.collections.new("Model")
                root_collection.children.link(model_collection)

                # Exclude the base collection that isn't transformed.
                # This prevents viewing and rendering in all modes.
                # This has to be done through the view layer instead of globally.
                bpy.context.view_layer.layer_collection.children[
                    root_collection.name
                ].children[model_collection.name].exclude = True

                for i, mesh in enumerate(model.meshes):
                    # Many materials are for meshes that won't be loaded.
                    # Lazy load materials to improve import times.
                    material = models.materials[mesh.material_index]
                    material_name = f"{mesh.material_index}.{material.name}"

                    blender_material = bpy.data.materials.get(material_name)
                    if blender_material is None:
                        blender_material = import_material(
                            material_name,
                            material,
                            blender_images,
                            root.image_textures,
                            models.samplers,
                        )

                    buffers = group.buffers[model.model_buffers_index]

                    if not import_all_meshes:
                        if (
                            base_lods is not None
                            and mesh.lod_item_index not in base_lods
                        ):
                            continue

                        if "_outline" in material.name or "_speff_" in material.name:
                            continue

                    import_mesh(
                        operator,
                        None,
                        model_collection,
                        buffers,
                        models,
                        mesh,
                        blender_material,
                        material.name,
                        flip_uvs,
                        i,
                        import_outlines=True,
                    )

                # Instances technically apply to the entire model.
                # Just instance each mesh for now for simplicity.
                for i, transform in enumerate(model.instances):
                    # Transform the instance using the in game coordinate system and convert back.
                    matrix_world = (
                        y_up_to_z_up
                        @ Matrix(transform).transposed()
                        @ y_up_to_z_up.inverted()
                    )

                    collection_instance = bpy.data.objects.new(
                        f"ModelInstance{i}", None
                    )
                    collection_instance.instance_type = "COLLECTION"
                    collection_instance.instance_collection = model_collection
                    collection_instance.matrix_world = matrix_world
                    root_collection.objects.link(collection_instance)


def import_model_root(
    operator,
    root,
    blender_images,
    root_obj,
    import_all_meshes: bool,
    import_outlines: bool,
    flip_uvs: bool,
):
    base_lods = None
    if root.models.lod_data is not None:
        base_lods = [g.base_lod_index for g in root.models.lod_data.groups]

    # TODO: Cache based on vertex and index buffer indices?
    for model in root.models.models:
        for i, mesh in enumerate(model.meshes):
            # Many materials are for meshes that won't be loaded.
            # Lazy load materials to improve import times.
            material = root.models.materials[mesh.material_index]
            material_name = f"{mesh.material_index}.{material.name}"

            blender_material = bpy.data.materials.get(material_name)
            if blender_material is None:
                blender_material = import_material(
                    material_name,
                    material,
                    blender_images,
                    root.image_textures,
                    root.models.samplers,
                )

            if not import_all_meshes:
                if base_lods is not None and mesh.lod_item_index not in base_lods:
                    continue

                if "_outline" in material.name or "_speff_" in material.name:
                    continue

            import_mesh(
                operator,
                root_obj,
                bpy.context.collection,
                root.buffers,
                root.models,
                mesh,
                blender_material,
                material.name,
                flip_uvs,
                i,
                import_outlines,
            )


def import_mesh(
    operator,
    root_obj: Optional[bpy.types.Object],
    collection: bpy.types.Collection,
    buffers,
    models,
    mesh,
    material: bpy.types.Material,
    material_name: str,
    flip_uvs: bool,
    i: int,
    import_outlines: bool,
):
    blender_mesh = bpy.data.meshes.new(f"{i}.{material_name}")

    # Vertex buffers are shared with multiple index buffers.
    # In practice, only a small range of vertices are used.
    # Reindex the vertices to eliminate most loose vertices.
    index_buffer = buffers.index_buffers[mesh.index_buffer_index]
    min_index = 0
    max_index = 0
    if index_buffer.indices.size > 0:
        min_index = index_buffer.indices.min()
        max_index = index_buffer.indices.max()

    indices = index_buffer.indices.astype(np.uint32) - min_index
    loop_start = np.arange(0, indices.shape[0], 3, dtype=np.uint32)
    loop_total = np.full(loop_start.shape[0], 3, dtype=np.uint32)

    blender_mesh.loops.add(indices.shape[0])
    blender_mesh.loops.foreach_set("vertex_index", indices)

    blender_mesh.polygons.add(loop_start.shape[0])
    blender_mesh.polygons.foreach_set("loop_start", loop_start)
    blender_mesh.polygons.foreach_set("loop_total", loop_total)

    # Set vertex attributes.
    # TODO: Set remaining attributes
    vertex_buffer = buffers.vertex_buffers[mesh.vertex_buffer_index]

    # Don't assume the first attribute is position to properly set vertex count.
    for attribute in vertex_buffer.attributes:
        data = attribute.data[min_index : max_index + 1]

        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Position:
            blender_mesh.vertices.add(data.shape[0])
            blender_mesh.vertices.foreach_set("co", data.reshape(-1))

    morph_blend_positions = None
    for attribute in vertex_buffer.morph_blend_target:
        data = attribute.data[min_index : max_index + 1]

        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Position2:
            # Shape keys do their own indexing with the full data.
            morph_blend_positions = attribute.data

            # TODO: Don't assume the first attribute is position to set count.
            blender_mesh.vertices.add(data.shape[0])
            blender_mesh.vertices.foreach_set("co", data.reshape(-1))

    # Set attributes now that the vertices are added by the position attribute.
    for attribute in vertex_buffer.attributes:
        data = attribute.data[min_index : max_index + 1]

        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord0:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord0", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord1:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord1", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord2:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord2", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord3:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord3", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord4:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord4", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord5:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord5", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord6:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord6", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord7:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord7", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord8:
            import_uvs(operator, blender_mesh, indices, data, "TexCoord8", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.VertexColor:
            import_colors(blender_mesh, data, "VertexColor")
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.Blend:
            import_colors(blender_mesh, data, "Blend")

    outline_vertex_colors = None
    if vertex_buffer.outline_buffer_index is not None:
        outline_buffer = buffers.outline_buffers[vertex_buffer.outline_buffer_index]
        for a in outline_buffer.attributes:
            # Use a unique name to avoid conflicting with existing attributes.
            if a.attribute_type == xc3_model_py.vertex.AttributeType.VertexColor:
                data = a.data[min_index : max_index + 1]
                import_colors(blender_mesh, data, "OutlineVertexColor")
                outline_vertex_colors = data

    blender_mesh.update()

    # The validate call may modify and reindex geometry.
    # Assign normals now that the mesh has been updated.
    for attribute in vertex_buffer.attributes:
        if attribute.attribute_type in [
            xc3_model_py.vertex.AttributeType.Normal,
            xc3_model_py.vertex.AttributeType.Normal2,
        ]:
            # We can't assume that the attribute data is normalized.
            data = attribute.data[min_index : max_index + 1, :3]
            normals = normalize(data)
            blender_mesh.normals_split_custom_set_from_vertices(normals)

    for attribute in vertex_buffer.morph_blend_target:
        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Normal4:
            # We can't assume that the attribute data is normalized.
            data = attribute.data[min_index : max_index + 1, :3]
            normals = normalize(data)
            blender_mesh.normals_split_custom_set_from_vertices(normals)

    blender_mesh.validate()

    # Assign materials from the current group.
    blender_mesh.materials.append(material)

    # Convert from Y up to Z up.
    y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")
    blender_mesh.transform(y_up_to_z_up)

    obj = bpy.data.objects.new(blender_mesh.name, blender_mesh)

    # TODO: Is there a way to not do this for every instance?
    # Only non instanced character meshes are skinned in practice.
    if buffers.weights is not None:
        # Calculate the index offset based on the weight group for this mesh.
        pass_type = models.materials[mesh.material_index].pass_type
        lod_item_index = 0 if mesh.lod_item_index is None else mesh.lod_item_index
        start_index = buffers.weights.weights_start_index(
            mesh.flags2, lod_item_index, pass_type
        )

        # An extra step is required since some Xenoblade X models have multiple weight buffers.
        weight_buffer = buffers.weights.weight_buffer(mesh.flags2)
        if weight_buffer is not None:
            import_weight_groups(
                weight_buffer, start_index, obj, vertex_buffer, min_index, max_index
            )

    if len(vertex_buffer.morph_targets) > 0:
        import_shape_keys(
            vertex_buffer,
            models.morph_controller_names,
            morph_blend_positions,
            min_index,
            max_index,
            obj,
        )

    if import_outlines and vertex_buffer.outline_buffer_index is not None:
        # The vertex alpha controls the thickness.
        # The solidify modifier can only use vertex groups.
        # Vertex groups are also easier to paint and modify in Blender.
        # TODO: Is there a faster way than setting weights per vertex?
        group = obj.vertex_groups.new(name="OutlineThickness")
        if outline_vertex_colors is not None:
            for i in range(outline_vertex_colors.shape[0]):
                group.add([i], outline_vertex_colors[i, 3], "REPLACE")

        # TODO: Outline attributes for color and vertex group for thickness.
        # TODO: Find and use the original outline material if present.
        # TODO: Update the shader database to properly support outline materials.
        outline_material = create_outline_material()
        blender_mesh.materials.append(outline_material)

        # Create outlines using a single mesh and a modifier.
        # The outline data can be regenerated if needed for export.
        modifier = obj.modifiers.new(name="Solidify Outlines", type="SOLIDIFY")
        modifier.use_rim = False
        modifier.use_flip_normals = True
        modifier.material_offset = 1
        modifier.thickness = -0.0015
        modifier.vertex_group = "OutlineThickness"
        # Prevent rendering issues with overlapping geoemtry.
        # The in game shaders discard fragments with 0.0 outline alpha.
        modifier.thickness_vertex_group = 0.0001

    # Attach the mesh to the armature or empty.
    # Assume the root_obj is an armature if there are weights.
    # TODO: Find a more reliable way of checking this.
    obj.parent = root_obj
    if buffers.weights is not None:
        modifier = obj.modifiers.new(root_obj.data.name, type="ARMATURE")
        modifier.object = root_obj

    collection.objects.link(obj)


def create_outline_material():
    outline_material = bpy.data.materials.new("outlines")
    outline_material.use_nodes = True
    outline_material.use_backface_culling = True

    nodes = outline_material.node_tree.nodes
    links = outline_material.node_tree.links

    # Create the nodes from scratch to ensure the required nodes are present.
    # This avoids hard coding names like "Material Output" that depend on the UI language.
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (-100, 0)

    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.location = (-300, 0)
    vertex_color.layer_name = "OutlineVertexColor"
    links.new(vertex_color.outputs["Color"], emission.inputs["Color"])

    # Workaround to make inverted hull outlines work in cycles.
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-100, -150)

    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-100, 400)

    mix1 = nodes.new("ShaderNodeMixShader")
    mix1.location = (100, 0)
    links.new(geometry.outputs["Backfacing"], mix1.inputs["Fac"])
    links.new(emission.outputs["Emission"], mix1.inputs[1])
    links.new(transparent.outputs["BSDF"], mix1.inputs[2])

    light_path = nodes.new("ShaderNodeLightPath")
    light_path.location = (100, 400)

    mix2 = nodes.new("ShaderNodeMixShader")
    mix2.location = (300, 0)
    links.new(light_path.outputs["Is Camera Ray"], mix2.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix2.inputs[1])
    links.new(mix1.outputs["Shader"], mix2.inputs[2])

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (500, 0)
    links.new(mix2.outputs["Shader"], output_node.inputs["Surface"])

    return outline_material


def normalize(data: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(data, ord=2, axis=1)
    # Prevent divide by zero.
    lengths[lengths == 0] = 1.0
    return data / lengths.reshape((-1, 1))


def import_shape_keys(
    vertex_buffer, names: list[str], position_data, min_index: int, max_index: int, obj
):
    # Shape keys need to be relative to something.
    obj.shape_key_add(name="Basis")

    if position_data is None:
        return

    z_up_to_y_up = np.array(Matrix.Rotation(math.radians(-90), 3, "X"))

    for target in vertex_buffer.morph_targets:
        sk = obj.shape_key_add(name=names[target.morph_controller_index])
        if target.vertex_indices.size > 0:
            # Morph targets are stored as sparse deltas for the base positions.
            # TODO: Blender doesn't have shape key normals?
            positions = position_data.copy()
            positions[target.vertex_indices] += target.position_deltas

            # Account for the unused vertex removal performed for other attributes.
            final_positions = positions[min_index : max_index + 1] @ z_up_to_y_up
            sk.points.foreach_set("co", final_positions.reshape(-1))


def import_uvs(
    operator,
    blender_mesh: bpy.types.Mesh,
    vertex_indices: np.ndarray,
    data: np.ndarray,
    name: str,
    flip_uvs: bool,
):
    uv_layer = blender_mesh.uv_layers.new(name=name)

    if uv_layer is not None:
        # This is set per loop rather than per vertex.
        loop_uvs = data[vertex_indices]
        if flip_uvs:
            # Flip vertically to match Blender.
            loop_uvs[:, 1] = 1.0 - loop_uvs[:, 1]
        uv_layer.data.foreach_set("uv", loop_uvs.reshape(-1))
    else:
        # Blender has a limit of 8 UV maps.
        message = f"Skipping {name} for mesh {blender_mesh.name} to avoid exceeding UV limit of 8"
        operator.report({"WARNING"}, message)


def import_colors(
    blender_mesh: bpy.types.Mesh,
    data: np.ndarray,
    name: str,
):
    # Byte color still uses floats but restricts their range to 0.0 to 1.0.
    attribute = blender_mesh.color_attributes.new(
        name=name, type="BYTE_COLOR", domain="POINT"
    )
    attribute.data.foreach_set("color", data.reshape(-1))


def import_weight_groups(
    skin_weights,
    start_index: int,
    blender_mesh,
    vertex_buffer,
    min_index: int,
    max_index: int,
):
    # Find the per vertex skinning information.
    weight_indices = None
    for attribute in vertex_buffer.attributes:
        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.WeightIndex:
            # Account for adjusting vertex indices in a previous step.
            indices = attribute.data[min_index : max_index + 1]
            weight_indices = indices + start_index
            break

    if weight_indices is not None:
        # This automatically removes zero weights.
        influences = skin_weights.to_influences(weight_indices)

        for influence in influences:
            # Lazily load only used vertex groups.
            name = influence.bone_name
            group = blender_mesh.vertex_groups.get(name)
            if group is None:
                group = blender_mesh.vertex_groups.new(name=name)

                # TODO: Is there a faster way than setting weights per vertex?
                for weight in influence.weights:
                    group.add([weight.vertex_index], weight.weight, "REPLACE")


def import_material(name: str, material, blender_images, image_textures, samplers):
    blender_material = bpy.data.materials.new(name)

    # Add some custom properties to make debugging easier.
    blender_material["technique_index"] = material.technique_index

    blender_material.use_nodes = True
    nodes = blender_material.node_tree.nodes
    links = blender_material.node_tree.links

    # Create the nodes from scratch to ensure the required nodes are present.
    # This avoids hard coding names like "Material Output" that depend on the UI language.
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (200, 0)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (500, 0)

    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Get information on how the decompiled shader code assigns outputs.
    # The G-Buffer output textures can be mapped to inputs on the principled BSDF.
    # Textures provide less accurate fallback assignments based on usage hints.
    assignments = material.output_assignments(image_textures)
    mat_id = assignments.mat_id()
    assignments = assignments.assignments

    textures = []
    textures_rgb = []
    textures_scale = []
    textures_uv = []
    for i, texture in enumerate(material.textures):
        location_y = 300 - i * 300
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.label = str(i)
        texture_node.width = 330
        texture_node.location = (-900, location_y)
        texture_node.image = blender_images[texture.image_texture_index]

        # TODO: Use the full mat2x4 transform.
        scale = nodes.new("ShaderNodeVectorMath")
        scale.location = (-1100, location_y)
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = (1.0, 1.0, 1.0)
        textures_scale.append(scale)

        uv = nodes.new("ShaderNodeUVMap")
        uv.location = (-1300, location_y)
        uv.uv_map = "TexCoord0"
        textures_uv.append(uv)

        links.new(uv.outputs["UV"], scale.inputs["Vector"])
        links.new(scale.outputs["Vector"], texture_node.inputs["Vector"])

        # TODO: Check if U and V have the same address mode.
        try:
            sampler = samplers[texture.sampler_index]
            if sampler.address_mode_u == xc3_model_py.AddressMode.ClampToEdge:
                texture_node.extension = "CLIP"
            elif sampler.address_mode_u == xc3_model_py.AddressMode.Repeat:
                texture_node.extension = "REPEAT"
            elif sampler.address_mode_u == xc3_model_py.AddressMode.MirrorRepeat:
                texture_node.extension = "MIRROR"
        except:
            # TODO: Fix samplers for xcx models.
            pass

        textures.append(texture_node)

        texture_rgb_node = nodes.new("ShaderNodeSeparateColor")
        texture_rgb_node.location = (-500, location_y)
        textures_rgb.append(texture_rgb_node)
        links.new(texture_node.outputs["Color"], texture_rgb_node.inputs["Color"])

    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.location = (-710, 500)
    vertex_color.layer_name = "VertexColor"

    vertex_color_rgb = nodes.new("ShaderNodeSeparateColor")
    vertex_color_rgb.location = (-500, 500)
    links.new(vertex_color.outputs["Color"], vertex_color_rgb.inputs["Color"])

    vertex_color_nodes = (vertex_color_rgb, vertex_color)
    texture_nodes = (textures, textures_rgb, textures_scale, textures_uv)

    # TODO: Alpha testing.
    # TODO: Select UV map for each texture.
    # Assume the color texture isn't used as non color data.
    base_color = nodes.new("ShaderNodeCombineColor")
    base_color.location = (-200, 200)
    assign_channel(
        assignments[0].x,
        "x",
        links,
        texture_nodes,
        vertex_color_nodes,
        base_color.inputs["Red"],
        is_data=False,
    )
    assign_channel(
        assignments[0].y,
        "y",
        links,
        texture_nodes,
        vertex_color_nodes,
        base_color.inputs["Green"],
        is_data=False,
    )
    assign_channel(
        assignments[0].z,
        "z",
        links,
        texture_nodes,
        vertex_color_nodes,
        base_color.inputs["Blue"],
        is_data=False,
    )

    if material.state_flags.blend_mode not in [
        xc3_model_py.BlendMode.Disabled,
        xc3_model_py.BlendMode.Disabled2,
    ]:
        assign_channel(
            assignments[0].w,
            "w",
            links,
            texture_nodes,
            vertex_color_nodes,
            bsdf.inputs["Alpha"],
            is_data=False,
        )

    mix_ao = nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mix_ao.inputs["Factor"].default_value = 1.0
    mix_ao.inputs["B"].default_value = (1.0, 1.0, 1.0, 1.0)

    # Single channel ambient occlusion.
    assign_channel(
        assignments[2].z,
        "z",
        links,
        texture_nodes,
        vertex_color_nodes,
        mix_ao.inputs["B"],
    )

    if (
        assignments[0].x is None
        and assignments[0].y is None
        and assignments[0].z is None
    ):
        # TODO: multiply by gMatCol instead?
        # TODO: more accurate gamma handling
        mix_ao.inputs["A"].default_value = [c**2.2 for c in material.color]
    else:
        links.new(base_color.outputs["Color"], mix_ao.inputs["A"])

    if material.state_flags.blend_mode == xc3_model_py.BlendMode.Multiply:
        # Workaround for Blender not supporting alpha blending modes.
        transparent_bsdf = nodes.new("ShaderNodeBsdfTransparent")
        transparent_bsdf.location = (300, 100)
        links.new(mix_ao.outputs["Result"], transparent_bsdf.inputs["Color"])
        links.new(transparent_bsdf.outputs["BSDF"], output_node.inputs["Surface"])

        blender_material.blend_method = "BLEND"
    else:
        links.new(mix_ao.outputs["Result"], bsdf.inputs["Base Color"])

    assign_normal_map(
        nodes,
        links,
        bsdf,
        assignments,
        texture_nodes,
        vertex_color_nodes,
    )

    assign_channel(
        assignments[1].x,
        "x",
        links,
        texture_nodes,
        vertex_color_nodes,
        bsdf.inputs["Metallic"],
    )

    if (
        assignments[5].x is not None
        or assignments[5].y is not None
        or assignments[5].z is not None
    ):
        color = nodes.new("ShaderNodeCombineColor")
        color.location = (-200, -400)
        if mat_id in [2, 5] or mat_id is None:
            color.inputs["Red"].default_value = 1.0
            color.inputs["Green"].default_value = 1.0
            color.inputs["Blue"].default_value = 1.0

        assign_channel(
            assignments[5].x,
            "x",
            links,
            texture_nodes,
            vertex_color_nodes,
            color.inputs["Red"],
            is_data=False,
        )
        assign_channel(
            assignments[5].y,
            "y",
            links,
            texture_nodes,
            vertex_color_nodes,
            color.inputs["Green"],
            is_data=False,
        )
        assign_channel(
            assignments[5].z,
            "z",
            links,
            texture_nodes,
            vertex_color_nodes,
            color.inputs["Blue"],
            is_data=False,
        )

        # TODO: Toon and hair shaders always use specular color?
        # Xenoblade X models typically use specular but don't have a mat id value yet.
        # TODO: use the material render flags instead for better accuracy.
        if mat_id in [2, 5] or mat_id is None:
            links.new(color.outputs["Color"], bsdf.inputs["Specular Tint"])
        else:
            links.new(color.outputs["Color"], bsdf.inputs["Emission Color"])
            bsdf.inputs["Emission Strength"].default_value = 1.0

    # Invert glossiness to get roughness.
    if assignments[1].y is not None:
        value = assignments[1].y.value()
        if value is not None:
            bsdf.inputs["Roughness"].default_value = 1.0 - value
        else:
            invert = nodes.new("ShaderNodeMath")
            invert.location = (-200, 0)
            invert.operation = "SUBTRACT"
            invert.inputs[0].default_value = 1.0
            assign_channel(
                assignments[1].y,
                "y",
                links,
                texture_nodes,
                vertex_color_nodes,
                invert.inputs[1],
            )
            links.new(invert.outputs["Value"], bsdf.inputs["Roughness"])

    if material.alpha_test is not None:
        texture = material.alpha_test
        channel = ["Red", "Green", "Blue", "Alpha"][texture.channel_index]
        if channel == "Alpha":
            input = textures[texture.texture_index].outputs["Alpha"]
        else:
            input = textures_rgb[texture.texture_index].outputs[channel]
        links.new(input, bsdf.inputs["Alpha"])

        # TODO: Support alpha blending?
        blender_material.blend_method = "CLIP"
        blender_material.shadow_method = "CLIP"

    return blender_material


def create_node_group(nodes, name: str, create_node_tree):
    # Cache the node group creation.
    node_tree = bpy.data.node_groups.get(name)
    if node_tree is None:
        node_tree = create_node_tree()

    group = nodes.new("ShaderNodeGroup")
    group.node_tree = node_tree
    return group


def assign_normal_map(
    nodes, links, bsdf, assignments, texture_nodes, vertex_color_nodes
):
    if assignments[2].x is None and assignments[2].y is None:
        return

    normals_x = -200
    normals_y = -800

    base_normals = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
    base_normals.location = (-200, normals_y)
    normals_x += 200
    normals_y -= 200

    base_normals.inputs["X"].default_value = 0.5
    base_normals.inputs["Y"].default_value = 0.5

    assign_channel(
        assignments[2].x,
        "x",
        links,
        texture_nodes,
        vertex_color_nodes,
        base_normals.inputs["X"],
    )
    assign_channel(
        assignments[2].y,
        "y",
        links,
        texture_nodes,
        vertex_color_nodes,
        base_normals.inputs["Y"],
    )

    final_normals = base_normals

    for layer in assignments[2].layers:
        # TODO: handle remaining blend modes.
        if layer.blend_mode == xc3_model_py.shader_database.LayerBlendMode.Mix:
            mix_normals = create_node_group(nodes, "AddNormals", add_normals_node_group)
            mix_normals = nodes.new("ShaderNodeMix")
            mix_normals.data_type = "VECTOR"
        else:
            mix_normals = create_node_group(nodes, "AddNormals", add_normals_node_group)
        mix_normals.location = (normals_x, -800)
        normals_x += 200

        n2_normals = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
        n2_normals.inputs["X"].default_value = 0.5
        n2_normals.inputs["Y"].default_value = 0.5
        n2_normals.location = (-200, normals_y)
        normals_y -= -200
        links.new(n2_normals.outputs["Normal"], mix_normals.inputs["B"])

        assign_channel(
            layer.x,
            "x",
            links,
            texture_nodes,
            vertex_color_nodes,
            n2_normals.inputs["X"],
        )
        assign_channel(
            layer.y,
            "y",
            links,
            texture_nodes,
            vertex_color_nodes,
            n2_normals.inputs["Y"],
        )

        # TODO: Should this always assign the X channel?
        assign_channel(
            layer.weight,
            "x",
            links,
            texture_nodes,
            vertex_color_nodes,
            mix_normals.inputs["Factor"],
        )

        # Connect each add group to the next.
        if "Normal" in final_normals.outputs:
            links.new(final_normals.outputs["Normal"], mix_normals.inputs["A"])
        else:
            links.new(final_normals.outputs["Result"], mix_normals.inputs["A"])

        final_normals = mix_normals

    remap_normals = nodes.new("ShaderNodeVectorMath")
    remap_normals.location = (normals_x, -800)
    normals_x += 200
    remap_normals.operation = "MULTIPLY_ADD"
    if "Normal" in final_normals.outputs:
        links.new(final_normals.outputs["Normal"], remap_normals.inputs[0])
    else:
        links.new(final_normals.outputs["Result"], remap_normals.inputs[0])
    remap_normals.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap_normals.inputs[2].default_value = (0.5, 0.5, 0.5)

    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.location = (normals_x, -800)
    links.new(remap_normals.outputs["Vector"], normal_map.inputs["Color"])

    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])


def normals_xy_node_group():
    node_tree = bpy.data.node_groups.new("NormalsXY", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Normal"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-1400, 0)
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )

    remap_x = nodes.new("ShaderNodeMath")
    remap_x.location = (-1200, 100)
    remap_x.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["X"], remap_x.inputs[0])
    remap_x.inputs[1].default_value = 2.0
    remap_x.inputs[2].default_value = -1.0

    remap_y = nodes.new("ShaderNodeMath")
    remap_y.location = (-1200, -100)
    remap_y.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["Y"], remap_y.inputs[0])
    remap_y.inputs[1].default_value = 2.0
    remap_y.inputs[2].default_value = -1.0

    normal_xy = nodes.new("ShaderNodeCombineXYZ")
    normal_xy.location = (-1000, 0)
    links.new(remap_x.outputs["Value"], normal_xy.inputs["X"])
    links.new(remap_y.outputs["Value"], normal_xy.inputs["Y"])
    normal_xy.inputs["Z"].default_value = 0.0

    length2 = nodes.new("ShaderNodeVectorMath")
    length2.location = (-800, 0)
    length2.operation = "DOT_PRODUCT"
    links.new(normal_xy.outputs["Vector"], length2.inputs[0])
    links.new(normal_xy.outputs["Vector"], length2.inputs[1])

    one_minus_length = nodes.new("ShaderNodeMath")
    one_minus_length.location = (-600, 0)
    one_minus_length.operation = "SUBTRACT"
    one_minus_length.inputs[0].default_value = 1.0
    links.new(length2.outputs["Value"], one_minus_length.inputs[1])

    length = nodes.new("ShaderNodeMath")
    length.location = (-400, 0)
    length.operation = "SQRT"
    links.new(one_minus_length.outputs["Value"], length.inputs[0])

    normal_xyz = nodes.new("ShaderNodeCombineXYZ")
    normal_xyz.location = (-200, 0)
    links.new(remap_x.outputs["Value"], normal_xyz.inputs["X"])
    links.new(remap_y.outputs["Value"], normal_xyz.inputs["Y"])
    links.new(length.outputs["Value"], normal_xyz.inputs["Z"])

    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (0, 0)
    links.new(normal_xyz.outputs["Vector"], output_node.inputs["Normal"])

    return node_tree


def add_normals_node_group():
    node_tree = bpy.data.node_groups.new("AddNormals", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Normal"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-1700, 0)
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Factor"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="A"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="B"
    )

    t = nodes.new("ShaderNodeVectorMath")
    t.location = (-1500, 0)
    t.operation = "ADD"
    t.inputs[1].default_value = (0.0, 0.0, 1.0)
    links.new(input_node.outputs["A"], t.inputs[0])

    u = nodes.new("ShaderNodeVectorMath")
    u.location = (-1500, -200)
    u.operation = "MULTIPLY"
    u.inputs[1].default_value = (-1.0, -1.0, 1.0)
    links.new(input_node.outputs["B"], u.inputs[0])

    dot_t_u = nodes.new("ShaderNodeVectorMath")
    dot_t_u.location = (-1200, 0)
    dot_t_u.operation = "DOT_PRODUCT"
    links.new(t.outputs["Vector"], dot_t_u.inputs[0])
    links.new(u.outputs["Vector"], dot_t_u.inputs[1])

    t_xyz = nodes.new("ShaderNodeSeparateXYZ")
    t_xyz.location = (-1200, -200)
    links.new(t.outputs["Vector"], t_xyz.inputs["Vector"])

    multiply_t = nodes.new("ShaderNodeVectorMath")
    multiply_t.location = (-1000, 0)
    multiply_t.operation = "MULTIPLY"
    links.new(dot_t_u.outputs["Value"], multiply_t.inputs[0])
    links.new(t.outputs["Vector"], multiply_t.inputs[1])

    multiply_u = nodes.new("ShaderNodeVectorMath")
    multiply_u.location = (-1000, -200)
    multiply_u.operation = "MULTIPLY"
    links.new(u.outputs["Vector"], multiply_u.inputs[0])
    links.new(t_xyz.outputs["Z"], multiply_u.inputs[1])

    r = nodes.new("ShaderNodeVectorMath")
    r.location = (-800, -200)
    r.operation = "SUBTRACT"
    links.new(multiply_t.outputs["Vector"], r.inputs[0])
    links.new(multiply_u.outputs["Vector"], r.inputs[1])

    normalize_r = nodes.new("ShaderNodeVectorMath")
    normalize_r.location = (-600, -200)
    normalize_r.operation = "NORMALIZE"
    links.new(r.outputs["Vector"], normalize_r.inputs[0])

    mix = nodes.new("ShaderNodeMix")
    mix.location = (-400, 0)
    mix.data_type = "VECTOR"
    links.new(input_node.outputs["Factor"], mix.inputs["Factor"])
    links.new(input_node.outputs["A"], mix.inputs["A"])
    links.new(normalize_r.outputs["Vector"], mix.inputs["B"])

    normalize_result = nodes.new("ShaderNodeVectorMath")
    normalize_result.location = (-200, 0)
    normalize_result.operation = "NORMALIZE"
    links.new(mix.outputs["Result"], normalize_result.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (0, 0)
    links.new(normalize_result.outputs["Vector"], output_node.inputs["Normal"])

    return node_tree


def assign_channel(
    channel_assignment,
    output_channel,
    links,
    texture_nodes,
    vertex_color_nodes,
    output,
    is_data=True,
):
    vertex_color_rgb, vertex_color = vertex_color_nodes

    # Assign one output channel.
    if channel_assignment is not None:
        texture_assignment = channel_assignment.texture()
        value = channel_assignment.value()
        attribute = channel_assignment.attribute()

        # Values or attributes are assigned directly in shaders and should take priority.
        if value is not None:
            try:
                output.default_value = value
            except:
                output.default_value = (value, value, value, 1.0)
        elif attribute is not None:
            # TODO: Handle other attributes.
            if attribute.name == "vColor":
                channel_index = attribute.channel_index
                input_channel = ["Red", "Green", "Blue", "Alpha"][channel_index]

                # Alpha isn't part of the RGB node.
                if input_channel == "Alpha":
                    input = vertex_color.outputs["Alpha"]
                else:
                    input = vertex_color_rgb.outputs[input_channel]

                links.new(input, output)
        elif texture_assignment is not None:
            assign_texture_channel(
                texture_assignment, links, texture_nodes, output, is_data
            )


def assign_texture_channel(
    texture_assignment,
    links,
    texture_nodes,
    output,
    is_data=True,
):
    textures, textures_rgb, textures_scale, textures_uv = texture_nodes

    channel_index = "xyzw".index(texture_assignment.channels)
    input_channel = ["Red", "Green", "Blue", "Alpha"][channel_index]

    # Only handle sampler uniforms for material textures for now.
    sampler_to_index = {f"s{i}": i for i in range(10)}
    texture_index = sampler_to_index.get(texture_assignment.name)
    if texture_index is not None:
        try:
            # TODO: Find a better way to handle color management.
            # TODO: Why can't we just set everything to non color?
            # TODO: This won't work if users have different color spaces installed like aces.
            if is_data:
                textures[texture_index].image.colorspace_settings.name = "Non-Color"
            else:
                textures[texture_index].image.colorspace_settings.name = "sRGB"

                # Alpha isn't part of the RGB node.
            if input_channel == "Alpha":
                input = textures[texture_index].outputs["Alpha"]
            else:
                input = textures_rgb[texture_index].outputs[input_channel]

            links.new(input, output)

            for i in range(9):
                if texture_assignment.texcoord_name == f"vTex{i}":
                    textures_uv[texture_index].uv_map = f"TexCoord{i}"

                    # TODO: Create a node group for the mat2x4 transform (two dot products).
            if texture_assignment.texcoord_transforms is not None:
                transform_u, transform_v = texture_assignment.texcoord_transforms
                textures_scale[texture_index].inputs[1].default_value = (
                    transform_u[0],
                    transform_v[1],
                    1.0,
                )
        except IndexError:
            # TODO: Better error checking.
            print(f"Texture index {texture_index} out of range")
