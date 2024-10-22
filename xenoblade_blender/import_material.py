from typing import Dict, Optional
import bpy

from . import xc3_model_py


def import_material(
    name: str,
    material,
    blender_images: list[bpy.types.Image],
    shader_images: Dict[str, bpy.types.Image],
    image_textures,
    samplers,
):
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

    textures = {}
    textures_rgb = {}
    textures_scale = {}
    textures_uv = {}

    for i, texture in enumerate(material.textures):
        location_y = 300 - i * 300
        # TODO: xenoblade x samplers?
        sampler = None
        if texture.sampler_index < len(samplers):
            sampler = samplers[texture.sampler_index]
        add_texture_nodes(
            f"s{i}",
            blender_images[texture.image_texture_index],
            sampler,
            str(i),
            nodes,
            links,
            textures,
            textures_rgb,
            textures_scale,
            textures_uv,
            location_y,
        )
    for i, (name, image) in enumerate(shader_images.items()):
        # Place global textures after the material textures.
        # TODO: Don't load unused global textures?
        location_y = 300 - (i + len(material.textures)) * 300
        add_texture_nodes(
            name,
            image,
            None,
            name,
            nodes,
            links,
            textures,
            textures_rgb,
            textures_scale,
            textures_uv,
            location_y,
        )

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
        links,
        texture_nodes,
        vertex_color_nodes,
        base_color.inputs["Red"],
        is_data=False,
    )
    assign_channel(
        assignments[0].y,
        links,
        texture_nodes,
        vertex_color_nodes,
        base_color.inputs["Green"],
        is_data=False,
    )
    assign_channel(
        assignments[0].z,
        links,
        texture_nodes,
        vertex_color_nodes,
        base_color.inputs["Blue"],
        is_data=False,
    )

    if material.state_flags.blend_mode not in [
        xc3_model_py.material.BlendMode.Disabled,
        xc3_model_py.material.BlendMode.Disabled2,
    ]:
        assign_channel(
            assignments[0].w,
            links,
            texture_nodes,
            vertex_color_nodes,
            bsdf.inputs["Alpha"],
            is_data=False,
        )

    base_color_x = 0
    base_color_y = 400
    for layer_x, layer_y, layer_z in zip(
        assignments[0].x_layers, assignments[0].y_layers, assignments[0].z_layers
    ):
        # Assume the XYZ layers use the same values just with different channels.
        mix_color = mix_layer_values(
            layer_x,
            nodes,
            links,
            vertex_color_nodes,
            texture_nodes,
            (base_color_x, 400),
        )

        base_color_x += 200

        layer_value = nodes.new("ShaderNodeCombineColor")
        layer_value.location = (-200, base_color_y)
        base_color_y -= -200
        links.new(layer_value.outputs["Color"], mix_color.inputs["B"])

        assign_channel(
            layer_x.value,
            links,
            texture_nodes,
            vertex_color_nodes,
            layer_value.inputs["Red"],
        )
        assign_channel(
            layer_y.value,
            links,
            texture_nodes,
            vertex_color_nodes,
            layer_value.inputs["Green"],
        )
        assign_channel(
            layer_z.value,
            links,
            texture_nodes,
            vertex_color_nodes,
            layer_value.inputs["Blue"],
        )

        # Connect each layer to the next.
        if "Result" in base_color.outputs:
            links.new(base_color.outputs["Result"], mix_color.inputs["A"])
        else:
            links.new(base_color.outputs["Color"], mix_color.inputs["A"])

        base_color = mix_color

    mix_ao = nodes.new("ShaderNodeMix")
    mix_ao.location = (base_color_x, 400)
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mix_ao.inputs["Factor"].default_value = 1.0
    mix_ao.inputs["B"].default_value = (1.0, 1.0, 1.0, 1.0)

    # Single channel ambient occlusion.
    assign_channel(
        assignments[2].z,
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
        if "Result" in base_color.outputs:
            links.new(base_color.outputs["Result"], mix_ao.inputs["A"])
        else:
            links.new(base_color.outputs["Color"], mix_ao.inputs["A"])

    normal_map = assign_normal_map(
        nodes,
        links,
        bsdf,
        assignments,
        texture_nodes,
        vertex_color_nodes,
    )

    # Place the BSDF and output after all other nodes.
    if normal_map is not None:
        if normal_map.location[0] > bsdf.location[0]:
            bsdf.location[0] = normal_map.location[0] + 300
            output_node.location[0] = bsdf.location[0] + 300

    metallic_x = 0
    metallic_y = 100
    metallic = None
    for layer_x in assignments[1].x_layers:
        mix_metallic = mix_layer_values(
            layer_x,
            nodes,
            links,
            vertex_color_nodes,
            texture_nodes,
            (metallic_x, metallic_y),
        )
        metallic_x += 200

        # Assign the base layer.
        if metallic is None:
            assign_channel(
                assignments[1].x,
                links,
                texture_nodes,
                vertex_color_nodes,
                mix_metallic.inputs["A"],
            )

        assign_channel(
            layer_x.value,
            links,
            texture_nodes,
            vertex_color_nodes,
            mix_metallic.inputs["B"],
        )

        # Connect each layer to the next.
        if metallic is not None:
            if "Result" in metallic.outputs:
                links.new(metallic.outputs["Result"], mix_metallic.inputs["A"])
            else:
                links.new(metallic.outputs["Color"], mix_metallic.inputs["A"])

        metallic = mix_metallic

    # Place the BSDF and output after all other nodes.
    if metallic_x > bsdf.location[0]:
        bsdf.location[0] = metallic_x + 300
        output_node.location[0] = bsdf.location[0] + 300

    if metallic is not None:
        if "Result" in metallic.outputs:
            links.new(metallic.outputs["Result"], bsdf.inputs["Metallic"])
        else:
            links.new(metallic.outputs["Color"], bsdf.inputs["Metallic"])
    else:
        assign_channel(
            assignments[1].x,
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
            links,
            texture_nodes,
            vertex_color_nodes,
            color.inputs["Red"],
            is_data=False,
        )
        assign_channel(
            assignments[5].y,
            links,
            texture_nodes,
            vertex_color_nodes,
            color.inputs["Green"],
            is_data=False,
        )
        assign_channel(
            assignments[5].z,
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

    # TODO: layers for glossiness?
    # Invert glossiness to get roughness.
    if assignments[1].y is not None:
        value = assignments[1].y.value()
        if value is not None:
            bsdf.inputs["Roughness"].default_value = 1.0 - value
        else:
            invert = nodes.new("ShaderNodeMath")
            invert.location = (-200, -200)
            invert.operation = "SUBTRACT"
            invert.inputs[0].default_value = 1.0
            invert.label = "Glossiness"
            assign_channel(
                assignments[1].y,
                links,
                texture_nodes,
                vertex_color_nodes,
                invert.inputs[1],
            )
            links.new(invert.outputs["Value"], bsdf.inputs["Roughness"])

    final_albedo = mix_ao

    # Toon and hair materials use toon gradient ramps.
    if mat_id in [2, 5]:
        x_start = final_albedo.location[0] + 200
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.label = "gTToonGrad"
        texture_node.location = (x_start - 300, 900)
        if "gTToonGrad" in shader_images:
            texture_node.image = shader_images["gTToonGrad"]

        uvs = create_node_group(nodes, "ToonGradUVs", toon_grad_uvs_node_group)
        uvs.location = (x_start - 500, 900)
        links.new(uvs.outputs["Vector"], texture_node.inputs["Vector"])

        if normal_map is not None:
            links.new(normal_map.outputs["Normal"], uvs.inputs["Normal"])

        row_index = nodes.new("ShaderNodeValue")
        row_index.location = (x_start - 700, 900)
        row_index.label = "Toon Gradient Row"
        links.new(row_index.outputs["Value"], uvs.inputs["Row Index"])

        # Try and find the non processed value.
        # This works since type 26 only seems to be used for toon gradients.
        for c in material.work_callbacks:
            if c.unk1 == 26:
                row_index.outputs[0].default_value = material.work_values[c.unk2 + 1]
                break

        # Approximate the lighting ramp by multiplying base color.
        # This preserves compatibility with Cycles.
        mix_ramp = nodes.new("ShaderNodeMix")
        mix_ramp.location = (x_start, 900)
        mix_ramp.data_type = "RGBA"
        mix_ramp.blend_type = "MULTIPLY"
        mix_ramp.inputs["Factor"].default_value = 1.0
        links.new(texture_node.outputs["Color"], mix_ramp.inputs["A"])
        links.new(mix_ao.outputs["Result"], mix_ramp.inputs["B"])

        final_albedo = mix_ramp

    # Place the BSDF and output after all other nodes.
    if final_albedo.location[0] > bsdf.location[0]:
        bsdf.location[0] = final_albedo.location[0] + 300
        output_node.location[0] = bsdf.location[0] + 300

    if material.state_flags.blend_mode == xc3_model_py.material.BlendMode.Multiply:
        # Workaround for Blender not supporting alpha blending modes.
        transparent_bsdf = nodes.new("ShaderNodeBsdfTransparent")
        transparent_bsdf.location = (300, 100)
        links.new(final_albedo.outputs["Result"], transparent_bsdf.inputs["Color"])
        links.new(transparent_bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    else:
        links.new(final_albedo.outputs["Result"], bsdf.inputs["Base Color"])

    if material.state_flags.blend_mode not in [
        xc3_model_py.material.BlendMode.Disabled,
        xc3_model_py.material.BlendMode.Disabled2,
    ]:
        blender_material.blend_method = "BLEND"

    if material.alpha_test is not None:
        texture = material.alpha_test
        name = f"s{texture.texture_index}"
        channel = ["Red", "Green", "Blue", "Alpha"][texture.channel_index]
        if channel == "Alpha":
            input = textures[name].outputs["Alpha"]
        else:
            input = textures_rgb[name].outputs[channel]
        links.new(input, bsdf.inputs["Alpha"])

        # TODO: Support alpha blending?
        blender_material.blend_method = "CLIP"
        blender_material.shadow_method = "CLIP"

    # Remove unused global textures.
    # TODO: is there a better way of doing this?
    # TODO: Create a variable for this somewhere.
    for name in [
        "gTResidentTex09",
        "gTResidentTex43",
        "gTResidentTex44",
        "gTResidentTex45",
        "gTResidentTex46",
        "gTToonGrad",
    ]:
        node = textures_rgb.get(name)
        if node is not None:
            if all(len(o.links) == 0 for o in node.outputs):
                nodes.remove(textures[name])
                nodes.remove(textures_rgb[name])
                nodes.remove(textures_scale[name])
                nodes.remove(textures_uv[name])

    return blender_material


def mix_layer_values(layer, nodes, links, vertex_color_nodes, texture_nodes, location):
    match layer.blend_mode:
        case xc3_model_py.shader_database.LayerBlendMode.Mix:
            mix_values = nodes.new("ShaderNodeMix")
            mix_values.data_type = "RGBA"
        case xc3_model_py.shader_database.LayerBlendMode.MixRatio:
            mix_values = nodes.new("ShaderNodeMix")
            mix_values.data_type = "RGBA"
            mix_values.blend_type = "MULTIPLY"
        case xc3_model_py.shader_database.LayerBlendMode.Add:
            mix_values = nodes.new("ShaderNodeMix")
            mix_values.data_type = "RGBA"
            mix_values.blend_type = "ADD"
        case xc3_model_py.shader_database.LayerBlendMode.Overlay:
            mix_values = nodes.new("ShaderNodeMix")
            mix_values.data_type = "RGBA"
            mix_values.blend_type = "OVERLAY"
        case xc3_model_py.shader_database.LayerBlendMode.AddNormal:
            mix_values = create_node_group(nodes, "AddNormals", add_normals_node_group)
        case _:
            mix_values = nodes.new("ShaderNodeMix")
            mix_values.data_type = "RGBA"

    mix_values.location = location

    # TODO: Should this always assign the X channel?
    mix_values.inputs["Factor"].default_value = 0.0

    if layer.is_fresnel:
        fresnel_blend = create_node_group(
            nodes, "FresnelBlend", fresnel_blend_node_group
        )
        fresnel_blend.location = (
            location[0] - 200,
            location[1] + 200,
        )
        # TODO: normals?

        assign_channel(
            layer.weight,
            links,
            texture_nodes,
            vertex_color_nodes,
            fresnel_blend.inputs["Factor"],
        )
        links.new(fresnel_blend.outputs["Factor"], mix_values.inputs["Factor"])
    else:
        assign_channel(
            layer.weight,
            links,
            texture_nodes,
            vertex_color_nodes,
            mix_values.inputs["Factor"],
        )

    return mix_values


def toon_grad_uvs_node_group():
    node_tree = bpy.data.node_groups.new("ToonGradUVs", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-1000, 0)
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Row Index"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Normal"
    )

    uv = nodes.new("ShaderNodeCombineXYZ")
    uv.location = (-200, 150)

    # Using the actual lighting requires shader to RGB.
    # Use view based "lighting" to still support cycles.
    invert_facing = nodes.new("ShaderNodeMath")
    invert_facing.location = (-400, 150)
    invert_facing.operation = "SUBTRACT"
    invert_facing.inputs[0].default_value = 1.0
    links.new(invert_facing.outputs["Value"], uv.inputs["X"])

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.location = (-600, 150)
    layer_weight.inputs["Blend"].default_value = 0.5
    links.new(layer_weight.outputs["Facing"], invert_facing.inputs[1])
    links.new(input_node.outputs["Normal"], layer_weight.inputs["Normal"])

    flip_uvs = nodes.new("ShaderNodeMath")
    flip_uvs.location = (-400, 0)
    flip_uvs.label = "Flip UVs"
    flip_uvs.operation = "SUBTRACT"
    flip_uvs.inputs[0].default_value = 1.0
    links.new(flip_uvs.outputs["Value"], uv.inputs["Y"])

    div = nodes.new("ShaderNodeMath")
    div.location = (-600, 0)
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 256.0
    links.new(div.outputs["Value"], flip_uvs.inputs[1])

    add = nodes.new("ShaderNodeMath")
    add.location = (-800, 0)
    add.operation = "ADD"
    add.inputs[1].default_value = 0.5
    links.new(input_node.outputs["Row Index"], add.inputs[0])
    links.new(add.outputs["Value"], div.inputs[0])

    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (0, 0)
    links.new(uv.outputs["Vector"], output_node.inputs["Vector"])

    return node_tree


def add_texture_nodes(
    name,
    image,
    sampler,
    label,
    nodes,
    links,
    textures,
    textures_rgb,
    textures_scale,
    textures_uv,
    location_y,
):
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.label = label
    texture_node.width = 330
    texture_node.location = (-900, location_y)
    texture_node.image = image

    # TODO: Use the full mat2x4 transform.
    scale = nodes.new("ShaderNodeVectorMath")
    scale.location = (-1100, location_y)
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = (1.0, 1.0, 1.0)
    textures_scale[name] = scale

    uv = nodes.new("ShaderNodeUVMap")
    uv.location = (-1300, location_y)
    uv.uv_map = "TexCoord0"
    textures_uv[name] = uv

    links.new(uv.outputs["UV"], scale.inputs["Vector"])
    links.new(scale.outputs["Vector"], texture_node.inputs["Vector"])

    # TODO: Check if U and V have the same address mode.
    if sampler is not None:
        match sampler.address_mode_u:
            case xc3_model_py.AddressMode.ClampToEdge:
                texture_node.extension = "CLIP"
            case xc3_model_py.AddressMode.Repeat:
                texture_node.extension = "REPEAT"
            case xc3_model_py.AddressMode.MirrorRepeat:
                texture_node.extension = "MIRROR"

    textures[name] = texture_node

    texture_rgb_node = nodes.new("ShaderNodeSeparateColor")
    texture_rgb_node.location = (-500, location_y)
    textures_rgb[name] = texture_rgb_node
    links.new(texture_node.outputs["Color"], texture_rgb_node.inputs["Color"])


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
        links,
        texture_nodes,
        vertex_color_nodes,
        base_normals.inputs["X"],
    )
    assign_channel(
        assignments[2].y,
        links,
        texture_nodes,
        vertex_color_nodes,
        base_normals.inputs["Y"],
    )

    final_normals = base_normals

    for layer_x, layer_y in zip(assignments[2].x_layers, assignments[2].y_layers):
        # Assume the XY layers use the same values just with different channels.
        mix_normals = mix_layer_values(
            layer_x,
            nodes,
            links,
            vertex_color_nodes,
            texture_nodes,
            (normals_x, -800),
        )
        normals_x += 200

        n2_normals = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
        n2_normals.inputs["X"].default_value = 0.5
        n2_normals.inputs["Y"].default_value = 0.5
        n2_normals.location = (-200, normals_y)
        normals_y -= -200
        links.new(n2_normals.outputs["Normal"], mix_normals.inputs["B"])

        assign_channel(
            layer_x.value,
            links,
            texture_nodes,
            vertex_color_nodes,
            n2_normals.inputs["X"],
        )
        assign_channel(
            layer_y.value,
            links,
            texture_nodes,
            vertex_color_nodes,
            n2_normals.inputs["Y"],
        )

        # Connect each layer to the next.
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

    return normal_map


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


def fresnel_blend_node_group():
    node_tree = bpy.data.node_groups.new("FresnelBlend", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Factor"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-600, 0)
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Factor"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Normal"
    )

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.location = (-400, 0)
    links.new(input_node.outputs["Normal"], layer_weight.inputs["Normal"])

    multiply = nodes.new("ShaderNodeMath")
    multiply.location = (-400, -200)
    multiply.operation = "MULTIPLY"
    links.new(input_node.outputs["Factor"], multiply.inputs[0])
    multiply.inputs[1].default_value = 5.0

    pow_5 = nodes.new("ShaderNodeMath")
    pow_5.location = (-200, 0)
    pow_5.operation = "POWER"
    links.new(layer_weight.outputs["Facing"], pow_5.inputs[0])
    links.new(multiply.outputs["Value"], pow_5.inputs[1])

    output_node = nodes.new("NodeGroupOutput")
    output_node.location = (0, 0)
    links.new(pow_5.outputs["Value"], output_node.inputs["Factor"])

    return node_tree


def assign_channel(
    channel_assignment,
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
    # TODO: lazy load global textures?
    textures, textures_rgb, textures_scale, textures_uv = texture_nodes

    channel_index = "xyzw".index(texture_assignment.channels)
    input_channel = ["Red", "Green", "Blue", "Alpha"][channel_index]

    name = texture_assignment.name

    try:
        # TODO: Find a better way to handle color management.
        # TODO: Why can't we just set everything to non color?
        # TODO: This won't work if users have different color spaces installed like aces.
        if is_data:
            textures[name].image.colorspace_settings.name = "Non-Color"
        else:
            textures[name].image.colorspace_settings.name = "sRGB"

            # Alpha isn't part of the RGB node.
        if input_channel == "Alpha":
            input = textures[name].outputs["Alpha"]
        else:
            input = textures_rgb[name].outputs[input_channel]

        links.new(input, output)

        for i in range(9):
            if texture_assignment.texcoord_name == f"vTex{i}":
                textures_uv[name].uv_map = f"TexCoord{i}"

        # TODO: Create a node group for the mat2x4 transform (two dot products).
        if texture_assignment.texcoord_transforms is not None:
            transform_u, transform_v = texture_assignment.texcoord_transforms
            textures_scale[name].inputs[1].default_value = (
                transform_u[0],
                transform_v[1],
                1.0,
            )
    except KeyError:
        # TODO: Better error checking.
        print(f"Unable to assign texture {name}")
