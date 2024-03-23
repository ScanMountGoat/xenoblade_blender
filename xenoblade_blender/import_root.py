import bpy
import numpy as np
import os

from . import xc3_model_py
from mathutils import Matrix


def get_database_path(version: str) -> str:
    files = {'XC1': "xc1.json", 'XC2': "xc2.json", 'XC3': "xc3.json"}
    return os.path.join(os.path.dirname(__file__), files[version])


def import_armature(context, root, name: str):
    armature = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    armature.data.display_type = 'STICK'
    bpy.context.collection.objects.link(armature)

    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    if root.skeleton is not None:
        transforms = root.skeleton.model_space_transforms()

        for bone, transform in zip(root.skeleton.bones, transforms):
            new_bone = armature.data.edit_bones.new(name=bone.name)
            # TODO: Point bones towards their child?
            new_bone.head = [0, 0, 0]
            new_bone.tail = [0, 1, 0]
            new_bone.matrix = Matrix(transform).transposed()

        for bone in root.skeleton.bones:
            if bone.parent_index is not None:
                parent_bone_name = root.skeleton.bones[bone.parent_index].name
                parent_bone = armature.data.edit_bones.get(parent_bone_name)
                armature.data.edit_bones.get(bone.name).parent = parent_bone

    bpy.ops.object.mode_set(mode='OBJECT')

    return armature


def import_images(root):
    blender_images = []

    for image, decoded in zip(root.image_textures, root.decode_images_rgbaf32()):
        name = image.name if image.name is not None else 'image'
        blender_image = bpy.data.images.new(name, image.width, image.height)
        blender_image.pixels.foreach_set(decoded)
        # TODO: This should depend on srgb vs linear in format.
        # TODO: Why does this cause weird issues on saving?
        blender_image.colorspace_settings.is_data = True
        blender_images.append(blender_image)

    return blender_images


def import_root(root, blender_images, root_obj):
    for group in root.groups:
        for models in group.models:
            materials = import_materials(
                blender_images, models, root.image_textures)

            for model in models.models:
                for mesh in model.meshes:
                    # TODO: check actual base lod
                    # TODO: Include all meshes for proper exporting later?
                    material = materials[mesh.material_index]
                    if mesh.lod > 1 or "_outline" in material.name or "_speff_" in material.name:
                        continue
                    import_mesh(root_obj, group, models, model, mesh, material)


def import_mesh(root_obj, group, models, model, mesh, material):
    blender_mesh = bpy.data.meshes.new(material.name)

    buffers = group.buffers[model.model_buffers_index]
    # Vertex buffers are shared with multiple index buffers.
    # In practice, only a small range of vertices are used.
    # Reindex the vertices to eliminate most loose vertices.
    index_buffer = buffers.index_buffers[mesh.index_buffer_index]
    min_index = index_buffer.indices.min()
    max_index = index_buffer.indices.max()

    vertex_indices = index_buffer.indices.astype(np.uint32) - min_index
    loop_start = np.arange(0, vertex_indices.shape[0], 3, dtype=np.uint32)
    loop_total = np.full(loop_start.shape[0], 3, dtype=np.uint32)

    blender_mesh.loops.add(vertex_indices.shape[0])
    blender_mesh.loops.foreach_set('vertex_index', vertex_indices)

    blender_mesh.polygons.add(loop_start.shape[0])
    blender_mesh.polygons.foreach_set('loop_start', loop_start)
    blender_mesh.polygons.foreach_set('loop_total', loop_total)

    # Set vertex attributes.
    # TODO: Set remaining attributes
    # TODO: Helper functions for setting each attribute type.
    vertex_buffer = buffers.vertex_buffers[mesh.vertex_buffer_index]
    for attribute in vertex_buffer.attributes:
        data = attribute.data[min_index:max_index+1]

        if attribute.attribute_type == xc3_model_py.AttributeType.Position:
            # TODO: Don't assume the first attribute is position to set count.
            blender_mesh.vertices.add(data.shape[0])
            blender_mesh.vertices.foreach_set('co', data.reshape(-1))
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord0:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord0')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord1:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord1')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord2:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord2')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord3:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord3')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord4:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord4')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord5:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord5')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord6:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord6')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord7:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord7')
        elif attribute.attribute_type == xc3_model_py.AttributeType.TexCoord8:
            import_uvs(blender_mesh, vertex_indices, data, 'TexCoord8')
        elif attribute.attribute_type == xc3_model_py.AttributeType.VertexColor:
            import_colors(blender_mesh, vertex_indices, data, 'VertexColor')
        elif attribute.attribute_type == xc3_model_py.AttributeType.Blend:
            import_colors(blender_mesh, vertex_indices, data, 'Blend')

    # TODO: Will this mess up indexing for weight groups?
    blender_mesh.update()

    # The validate call may modify and reindex geometry.
    # Assign normals now that the mesh has been updated.
    for attribute in vertex_buffer.attributes:
        if attribute.attribute_type == xc3_model_py.AttributeType.Normal:
            # We can't assume that the attribute data is normalized.
            data = attribute.data[min_index:max_index+1, :3]
            lengths = np.linalg.norm(data, ord=2, axis=1)
            normals = data / lengths.reshape((-1, 1))

            # Auto smooth also enables custom vertex normals.
            blender_mesh.use_auto_smooth = True
            blender_mesh.normals_split_custom_set_from_vertices(normals)

    blender_mesh.validate()

    # Assign materials from the current group.
    blender_mesh.materials.append(material)

    # Instances technically apply to the entire model.
    # Just instance each mesh for now for simplicity.
    for transform in model.instances:
        obj = bpy.data.objects.new(
            blender_mesh.name, blender_mesh)
        obj.matrix_local = Matrix(
            transform).transposed()

        # TODO: Is there a way to not do this for every instance?
        # Only non instanced character meshes are skinned in practice.
        if buffers.weights is not None:
            # Calculate the index offset based on the weight group for this mesh.
            pass_type = models.materials[mesh.material_index].pass_type
            start_index = buffers.weights.weights_start_index(
                mesh.skin_flags, mesh.lod, pass_type)

            import_weight_groups(
                buffers.weights, start_index, obj, vertex_buffer, min_index, max_index)

        # Attach the mesh to the armature or empty.
        # Assume the root_obj is an armature if there are weights.
        # TODO: Find a more reliable way of checking this.
        obj.parent = root_obj
        if buffers.weights is not None:
            modifier = obj.modifiers.new(
                root_obj.data.name, type='ARMATURE')
            modifier.object = root_obj

        bpy.context.collection.objects.link(obj)


def import_uvs(blender_mesh: bpy.types.Mesh, vertex_indices: np.ndarray, data: np.ndarray, name: str):
    uv_layer = blender_mesh.uv_layers.new(name=name)
    # This is set per loop rather than per vertex.
    loop_uvs = data[vertex_indices].reshape(-1)
    uv_layer.data.foreach_set('uv', loop_uvs)


def import_colors(blender_mesh: bpy.types.Mesh, vertex_indices: np.ndarray, data: np.ndarray, name: str):
    # TODO: Just set this per vertex instead?
    # Byte color still uses floats but restricts their range to 0.0 to 1.0.
    attribute = blender_mesh.color_attributes.new(
        name=name, type='BYTE_COLOR', domain='CORNER')

    # This is set per loop rather than per vertex.
    loop_colors = data[vertex_indices].reshape(-1)
    attribute.data.foreach_set('color', loop_colors)


def import_weight_groups(weights, start_index: int, blender_mesh, vertex_buffer, min_index: int, max_index: int):
    # Find the per vertex skinning information.
    weight_indices = None
    for attribute in vertex_buffer.attributes:
        if attribute.attribute_type == xc3_model_py.AttributeType.WeightIndex:
            # Account for adjusting vertex indices in a previous step.
            weight_indices = attribute.data[min_index:max_index +
                                            1] + start_index
            break

    if weight_indices is not None:
        # This automatically removes zero weights.
        influences = weights.skin_weights.to_influences(weight_indices)

        for influence in influences:
            # Lazily load only used vertex groups.
            name = influence.bone_name
            group = blender_mesh.vertex_groups.get(name)
            if group is None:
                group = blender_mesh.vertex_groups.new(name=name)

                # TODO: Is there a faster way than setting weights per vertex?
                for weight in influence.weights:
                    group.add([weight.vertex_index], weight.weight, 'REPLACE')


def import_materials(blender_images, models, image_textures):
    materials = []
    for material in models.materials:
        blender_material = import_material(
            material, blender_images, image_textures)
        materials.append(blender_material)

    return materials


def import_material(material, blender_images, image_textures):
    blender_material = bpy.data.materials.new(material.name)
    blender_material.use_nodes = True

    nodes = blender_material.node_tree.nodes
    links = blender_material.node_tree.links

    bsdf = blender_material.node_tree.nodes["Principled BSDF"]

    # Get information on how the decompiled shader code assigns outputs.
    # The G-Buffer output textures can be mapped to inputs on the principled BSDF.
    # Textures provide less accurate fallback assignments based on usage hints.
    assignments = material.output_assignments(image_textures).assignments

    textures = []
    textures_rgb = []
    for texture in material.textures:
        texture_node = nodes.new('ShaderNodeTexImage')
        texture_node.image = blender_images[texture.image_texture_index]
        textures.append(texture_node)

        texture_rgb_node = nodes.new('ShaderNodeSeparateColor')
        textures_rgb.append(texture_rgb_node)
        links.new(texture_node.outputs['Color'],
                  texture_rgb_node.inputs['Color'])

    # TODO: Alpha testing.
    # TODO: Select UV map and scale for each texture.
    # TODO: Set color space for images?

    base_color = nodes.new('ShaderNodeCombineColor')
    assign_channel(assignments[0].x, links, textures,
                   textures_rgb, base_color, 'Red')
    assign_channel(assignments[0].y, links, textures,
                   textures_rgb, base_color, 'Green')
    assign_channel(assignments[0].z, links, textures,
                   textures_rgb, base_color, 'Blue')
    links.new(base_color.outputs['Color'], bsdf.inputs['Base Color'])

    assign_normal_map(nodes, links, bsdf, assignments, textures, textures_rgb)

    assign_channel(assignments[1].x, links, textures,
                   textures_rgb, bsdf, 'Metallic')

    # Invert glossiness to get roughness.
    invert = nodes.new('ShaderNodeMath')
    invert.operation = 'SUBTRACT'
    invert.inputs[0].default_value = 1.0
    assign_channel(assignments[1].y, links, textures, textures_rgb, invert, 1)
    links.new(invert.outputs['Value'], bsdf.inputs['Roughness'])

    if material.alpha_test is not None:
        texture = material.alpha_test
        channel = ['Red', 'Green', 'Blue', 'Alpha'][texture.channel_index]
        if channel == 'Alpha':
            input = textures[texture.texture_index].outputs['Alpha']
        else:
            input = textures_rgb[texture.texture_index].outputs[channel]
        links.new(input, bsdf.inputs['Alpha'])

        # TODO: Support alpha blending?
        blender_material.blend_method = 'CLIP'
        blender_material.shadow_method = 'CLIP'

    return blender_material


def assign_normal_map(nodes, links, bsdf, assignments, textures, textures_rgb):
    if assignments[2].x is None and assignments[2].y is None:
        return

    normal_xy = nodes.new('ShaderNodeCombineXYZ')
    assign_channel(assignments[2].x, links, textures,
                   textures_rgb, normal_xy, 'X')
    assign_channel(assignments[2].y, links, textures,
                   textures_rgb, normal_xy, 'Y')
    normal_xy.inputs['Z'].default_value = 0.0

    length2 = nodes.new('ShaderNodeVectorMath')
    length2.operation = 'DOT_PRODUCT'
    links.new(normal_xy.outputs['Vector'], length2.inputs[0])
    links.new(normal_xy.outputs['Vector'], length2.inputs[1])

    one_minus_length = nodes.new('ShaderNodeMath')
    one_minus_length.operation = 'SUBTRACT'
    one_minus_length.inputs[0].default_value = 1.0
    links.new(length2.outputs['Value'], one_minus_length.inputs[1])

    length = nodes.new('ShaderNodeMath')
    length.operation = 'SQRT'
    links.new(one_minus_length.outputs['Value'], length.inputs[0])

    normal_xyz = nodes.new('ShaderNodeCombineXYZ')
    assign_channel(assignments[2].x, links, textures,
                   textures_rgb, normal_xyz, 'X')
    assign_channel(assignments[2].y, links, textures,
                   textures_rgb, normal_xyz, 'Y')
    links.new(length.outputs['Value'], normal_xyz.inputs['Z'])

    normal_map = nodes.new('ShaderNodeNormalMap')
    links.new(normal_xyz.outputs['Vector'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])


def assign_channel(channel_assignment, links, textures, textures_rgb, output_node, output_channel):
    # Assign one output channel from a texture channel or constant.
    if channel_assignment is not None:
        texture_assignment = channel_assignment.texture()
        value = channel_assignment.value()

        if texture_assignment is not None:
            input_channels = ['Red', 'Green', 'Blue', 'Alpha']
            input_channel = input_channels[texture_assignment.channel_index]

            # Only handle sampler uniforms for material textures for now.
            sampler_to_index = {f's{i}': i for i in range(10)}
            texture_index = sampler_to_index.get(texture_assignment.name)
            if texture_index is not None:
                try:
                    # Alpha isn't part of the RGB node.
                    if input_channel == 'Alpha':
                        input = textures[texture_index].outputs['Alpha']
                    else:
                        input = textures_rgb[texture_index].outputs[input_channel]
                    output = output_node.inputs[output_channel]
                    links.new(input, output)

                    if texture_assignment.texcoord_scale is not None:
                        pass
                except IndexError:
                    # TODO: Better error checking.
                    print(f'Texture index {texture_index} out of range')
        elif value is not None:
            output_node.inputs[output_channel].default_value = value
