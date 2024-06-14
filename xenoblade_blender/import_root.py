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
        for i, (image, decoded) in enumerate(
            zip(root.image_textures, root.decode_images_rgbaf32())
        ):
            # Use the same naming conventions as the saved PNG images and xc3_tex.
            if image.name is not None:
                name = f"{model_name}.{i}.{image.name}"
            else:
                name = f"{model_name}.{i}"

            blender_image = bpy.data.images.new(name, image.width, image.height)

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
                        None,
                        model_collection,
                        buffers,
                        models,
                        mesh,
                        blender_material,
                        material.name,
                        flip_uvs,
                        i,
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
    root, blender_images, root_obj, import_all_meshes: bool, flip_uvs: bool
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
                root_obj,
                bpy.context.collection,
                root.buffers,
                root.models,
                mesh,
                blender_material,
                material.name,
                flip_uvs,
                i,
            )


def import_mesh(
    root_obj: Optional[bpy.types.Object],
    collection: bpy.types.Collection,
    buffers,
    models,
    mesh,
    material: bpy.types.Material,
    material_name: str,
    flip_uvs: bool,
    i: int,
):
    blender_mesh = bpy.data.meshes.new(f"{i}.{material_name}")

    # Vertex buffers are shared with multiple index buffers.
    # In practice, only a small range of vertices are used.
    # Reindex the vertices to eliminate most loose vertices.
    index_buffer = buffers.index_buffers[mesh.index_buffer_index]
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
    # TODO: Helper functions for setting each attribute type.
    vertex_buffer = buffers.vertex_buffers[mesh.vertex_buffer_index]
    for attribute in vertex_buffer.attributes:
        data = attribute.data[min_index : max_index + 1]

        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Position:
            # TODO: Don't assume the first attribute is position to set count.
            blender_mesh.vertices.add(data.shape[0])
            blender_mesh.vertices.foreach_set("co", data.reshape(-1))
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord0:
            import_uvs(blender_mesh, indices, data, "TexCoord0", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord1:
            import_uvs(blender_mesh, indices, data, "TexCoord1", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord2:
            import_uvs(blender_mesh, indices, data, "TexCoord2", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord3:
            import_uvs(blender_mesh, indices, data, "TexCoord3", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord4:
            import_uvs(blender_mesh, indices, data, "TexCoord4", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord5:
            import_uvs(blender_mesh, indices, data, "TexCoord5", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord6:
            import_uvs(blender_mesh, indices, data, "TexCoord6", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord7:
            import_uvs(blender_mesh, indices, data, "TexCoord7", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.TexCoord8:
            import_uvs(blender_mesh, indices, data, "TexCoord8", flip_uvs)
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.VertexColor:
            import_colors(blender_mesh, indices, data, "VertexColor")
        elif attribute.attribute_type == xc3_model_py.vertex.AttributeType.Blend:
            import_colors(blender_mesh, indices, data, "Blend")

    position_data = None
    for attribute in vertex_buffer.morph_blend_target:
        data = attribute.data[min_index : max_index + 1]

        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Position2:
            # Shape keys do their own indexing with the full data.
            position_data = attribute.data

            # TODO: Don't assume the first attribute is position to set count.
            blender_mesh.vertices.add(data.shape[0])
            blender_mesh.vertices.foreach_set("co", data.reshape(-1))

    # TODO: Will this mess up indexing for weight groups?
    blender_mesh.update()

    # The validate call may modify and reindex geometry.
    # Assign normals now that the mesh has been updated.
    for attribute in vertex_buffer.attributes:
        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Normal:
            # We can't assume that the attribute data is normalized.
            data = attribute.data[min_index : max_index + 1, :3]
            lengths = np.linalg.norm(data, ord=2, axis=1)
            # TODO: Prevent divide by zero?
            normals = data / lengths.reshape((-1, 1))
            blender_mesh.normals_split_custom_set_from_vertices(normals)

    for attribute in vertex_buffer.morph_blend_target:
        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Normal4:
            # We can't assume that the attribute data is normalized.
            data = attribute.data[min_index : max_index + 1, :3] * 2.0 - 1.0
            lengths = np.linalg.norm(data, ord=2, axis=1)
            # TODO: Prevent divide by zero?
            normals = data / lengths.reshape((-1, 1))
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

    # TODO: Is there a way to not do this for every instance?
    # Only non instanced character meshes have shape keys in practice.
    if len(vertex_buffer.morph_targets) > 0:
        import_shape_keys(
            vertex_buffer,
            models.morph_controller_names,
            position_data,
            min_index,
            max_index,
            obj,
        )

    # Attach the mesh to the armature or empty.
    # Assume the root_obj is an armature if there are weights.
    # TODO: Find a more reliable way of checking this.
    obj.parent = root_obj
    if buffers.weights is not None:
        modifier = obj.modifiers.new(root_obj.data.name, type="ARMATURE")
        modifier.object = root_obj

    collection.objects.link(obj)


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
    blender_mesh: bpy.types.Mesh,
    vertex_indices: np.ndarray,
    data: np.ndarray,
    name: str,
    flip_uvs: bool,
):
    uv_layer = blender_mesh.uv_layers.new(name=name)
    # This is set per loop rather than per vertex.
    loop_uvs = data[vertex_indices]
    if flip_uvs:
        # Flip vertically to match Blender.
        loop_uvs[:, 1] = 1.0 - loop_uvs[:, 1]
    uv_layer.data.foreach_set("uv", loop_uvs.reshape(-1))


def import_colors(
    blender_mesh: bpy.types.Mesh,
    vertex_indices: np.ndarray,
    data: np.ndarray,
    name: str,
):
    # TODO: Just set this per vertex instead?
    # Byte color still uses floats but restricts their range to 0.0 to 1.0.
    attribute = blender_mesh.color_attributes.new(
        name=name, type="BYTE_COLOR", domain="CORNER"
    )

    # This is set per loop rather than per vertex.
    loop_colors = data[vertex_indices].reshape(-1)
    attribute.data.foreach_set("color", loop_colors)


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
    blender_material.use_nodes = True

    nodes = blender_material.node_tree.nodes
    links = blender_material.node_tree.links

    bsdf = nodes["Principled BSDF"]
    bsdf.location = (0, 0)

    nodes["Material Output"].location = (300, 0)

    # Get information on how the decompiled shader code assigns outputs.
    # The G-Buffer output textures can be mapped to inputs on the principled BSDF.
    # Textures provide less accurate fallback assignments based on usage hints.
    assignments = material.output_assignments(image_textures)
    mat_id = assignments.mat_id()
    assignments = assignments.assignments

    textures = []
    textures_rgb = []
    for i, texture in enumerate(material.textures):
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.label = str(i)
        texture_node.location = (-800, 300 - i * 300)
        texture_node.image = blender_images[texture.image_texture_index]

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
        texture_rgb_node.location = (-500, 300 - i * 300)
        textures_rgb.append(texture_rgb_node)
        links.new(texture_node.outputs["Color"], texture_rgb_node.inputs["Color"])

    # TODO: Alpha testing.
    # TODO: Select UV map and scale for each texture.
    # TODO: Set color space for images?
    # Assume the color texture isn't used as non color data.
    base_color = nodes.new("ShaderNodeCombineColor")
    base_color.location = (-200, 200)
    assign_channel(
        assignments[0].x,
        links,
        textures,
        textures_rgb,
        base_color,
        "Red",
        is_data=False,
    )
    assign_channel(
        assignments[0].y,
        links,
        textures,
        textures_rgb,
        base_color,
        "Green",
        is_data=False,
    )
    assign_channel(
        assignments[0].z,
        links,
        textures,
        textures_rgb,
        base_color,
        "Blue",
        is_data=False,
    )
    links.new(base_color.outputs["Color"], bsdf.inputs["Base Color"])

    assign_normal_map(nodes, links, bsdf, assignments, textures, textures_rgb)

    assign_channel(assignments[1].x, links, textures, textures_rgb, bsdf, "Metallic")

    # TODO: toon and hair shaders always use specular color?
    if not mat_id in [2, 5]:
        # TODO: This doesn't always work?
        emi_color = nodes.new("ShaderNodeCombineColor")
        emi_color.location = (-200, -400)
        assign_channel(
            assignments[5].x,
            links,
            textures,
            textures_rgb,
            emi_color,
            "Red",
            is_data=False,
        )
        assign_channel(
            assignments[5].y,
            links,
            textures,
            textures_rgb,
            emi_color,
            "Green",
            is_data=False,
        )
        assign_channel(
            assignments[5].z,
            links,
            textures,
            textures_rgb,
            emi_color,
            "Blue",
            is_data=False,
        )
        links.new(emi_color.outputs["Color"], bsdf.inputs["Emission Color"])

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
            assign_channel(assignments[1].y, links, textures, textures_rgb, invert, 1)
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


def assign_normal_map(nodes, links, bsdf, assignments, textures, textures_rgb):
    if assignments[2].x is None and assignments[2].y is None:
        return

    # TODO: Group these nodes in a node group?
    normal_xy = nodes.new("ShaderNodeCombineXYZ")
    normal_xy.location = (-1200, -700)
    assign_channel(assignments[2].x, links, textures, textures_rgb, normal_xy, "X")
    assign_channel(assignments[2].y, links, textures, textures_rgb, normal_xy, "Y")
    normal_xy.inputs["Z"].default_value = 0.0

    length2 = nodes.new("ShaderNodeVectorMath")
    length2.location = (-1000, -700)
    length2.operation = "DOT_PRODUCT"
    links.new(normal_xy.outputs["Vector"], length2.inputs[0])
    links.new(normal_xy.outputs["Vector"], length2.inputs[1])

    one_minus_length = nodes.new("ShaderNodeMath")
    one_minus_length.location = (-800, -700)
    one_minus_length.operation = "SUBTRACT"
    one_minus_length.inputs[0].default_value = 1.0
    links.new(length2.outputs["Value"], one_minus_length.inputs[1])

    length = nodes.new("ShaderNodeMath")
    length.location = (-600, -700)
    length.operation = "SQRT"
    links.new(one_minus_length.outputs["Value"], length.inputs[0])

    normal_xyz = nodes.new("ShaderNodeCombineXYZ")
    normal_xyz.location = (-400, -700)
    assign_channel(assignments[2].x, links, textures, textures_rgb, normal_xyz, "X")
    assign_channel(assignments[2].y, links, textures, textures_rgb, normal_xyz, "Y")
    links.new(length.outputs["Value"], normal_xyz.inputs["Z"])

    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.location = (-200, -200)
    links.new(normal_xyz.outputs["Vector"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])


def assign_channel(
    channel_assignment,
    links,
    textures,
    textures_rgb,
    output_node,
    output_channel,
    is_data=True,
):
    # Assign one output channel from a texture channel or constant.
    if channel_assignment is not None:
        texture_assignment = channel_assignment.texture()
        value = channel_assignment.value()

        if texture_assignment is not None:
            input_channels = ["Red", "Green", "Blue", "Alpha"]
            input_channel = input_channels[texture_assignment.channel_index]

            # Only handle sampler uniforms for material textures for now.
            sampler_to_index = {f"s{i}": i for i in range(10)}
            texture_index = sampler_to_index.get(texture_assignment.name)
            if texture_index is not None:
                try:
                    # TODO: Find a better way to handle color management.
                    # TODO: Why can't we just set everything to non color?
                    # TODO: This won't work if users have different color spaces installed like aces.
                    if is_data:
                        textures[texture_index].image.colorspace_settings.name = (
                            "Non-Color"
                        )
                    else:
                        textures[texture_index].image.colorspace_settings.name = "sRGB"

                    # Alpha isn't part of the RGB node.
                    if input_channel == "Alpha":
                        input = textures[texture_index].outputs["Alpha"]
                    else:
                        input = textures_rgb[texture_index].outputs[input_channel]
                    output = output_node.inputs[output_channel]
                    links.new(input, output)

                    if texture_assignment.texcoord_scale is not None:
                        pass
                except IndexError:
                    # TODO: Better error checking.
                    print(f"Texture index {texture_index} out of range")
        elif value is not None:
            output_node.inputs[output_channel].default_value = value
