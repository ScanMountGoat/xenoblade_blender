from typing import Dict, Optional, Tuple
import bpy
import typing

from xenoblade_blender.node_layout import layout_nodes

if typing.TYPE_CHECKING:
    from ..xc3_model_py.xc3_model_py import xc3_model_py
else:
    from . import xc3_model_py


def import_material(
    name: str,
    material,
    blender_images: list[bpy.types.Image],
    shader_images: Dict[str, bpy.types.Image],
    image_textures,
    samplers,
) -> bpy.types.Material:
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

    output_node = nodes.new("ShaderNodeOutputMaterial")

    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Get information on how the decompiled shader code assigns outputs.
    # The G-Buffer output textures can be mapped to inputs on the principled BSDF.
    # Textures provide less accurate fallback assignments based on usage hints.
    output_assignments = material.output_assignments(image_textures)
    mat_id = output_assignments.mat_id()

    textures = material_images_samplers(material, blender_images, samplers)
    for name, image in shader_images.items():
        textures[name] = (image, None)

    # Assume the color texture isn't used as non color data.
    base_color = nodes.new("ShaderNodeCombineColor")
    assign_output(
        output_assignments.output_assignments[0].x,
        output_assignments.assignments,
        nodes,
        links,
        base_color.inputs["Red"],
        textures,
        is_data=False,
    )
    assign_output(
        output_assignments.output_assignments[0].y,
        output_assignments.assignments,
        nodes,
        links,
        base_color.inputs["Green"],
        textures,
        is_data=False,
    )
    assign_output(
        output_assignments.output_assignments[0].z,
        output_assignments.assignments,
        nodes,
        links,
        base_color.inputs["Blue"],
        textures,
        is_data=False,
    )

    if material.state_flags.blend_mode not in [
        xc3_model_py.material.BlendMode.Disabled,
        xc3_model_py.material.BlendMode.Disabled2,
    ]:
        assign_output(
            output_assignments.output_assignments[0].w,
            output_assignments.assignments,
            nodes,
            links,
            bsdf.inputs["Alpha"],
            textures,
            is_data=True,
        )

    mix_ao = nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    links.new(base_color.outputs["Color"], mix_ao.inputs["A"])
    mix_ao.inputs["B"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix_ao.inputs["Factor"].default_value = 1.0

    # Single channel ambient occlusion.
    assign_output(
        output_assignments.output_assignments[2].z,
        output_assignments.assignments,
        nodes,
        links,
        mix_ao.inputs["B"],
        textures,
        is_data=True,
    )

    normal_map = assign_normal_map(
        nodes,
        links,
        bsdf,
        output_assignments.output_assignments[2].x,
        output_assignments.output_assignments[2].y,
        output_assignments.normal_intensity,
        output_assignments.assignments,
        textures,
    )

    assign_output(
        output_assignments.output_assignments[1].x,
        output_assignments.assignments,
        nodes,
        links,
        bsdf.inputs["Metallic"],
        textures,
        is_data=True,
    )

    if (
        output_assignments.output_assignments[5].x is not None
        or output_assignments.output_assignments[5].y is not None
        or output_assignments.output_assignments[5].z is not None
    ):
        color = nodes.new("ShaderNodeCombineColor")
        if mat_id in [2, 5] or mat_id is None:
            color.inputs["Red"].default_value = 1.0
            color.inputs["Green"].default_value = 1.0
            color.inputs["Blue"].default_value = 1.0

        assign_output(
            output_assignments.output_assignments[5].x,
            output_assignments.assignments,
            nodes,
            links,
            color.inputs["Red"],
            textures,
            is_data=False,
        )
        assign_output(
            output_assignments.output_assignments[5].y,
            output_assignments.assignments,
            nodes,
            links,
            color.inputs["Green"],
            textures,
            is_data=False,
        )
        assign_output(
            output_assignments.output_assignments[5].z,
            output_assignments.assignments,
            nodes,
            links,
            color.inputs["Blue"],
            textures,
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
    invert = nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0
    assign_output(
        output_assignments.output_assignments[1].y,
        output_assignments.assignments,
        nodes,
        links,
        invert.inputs[1],
        textures,
        is_data=True,
    )
    links.new(invert.outputs["Value"], bsdf.inputs["Roughness"])

    final_albedo = mix_ao

    # Toon and hair materials use toon gradient ramps.
    if mat_id in [2, 5]:
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.label = "gTToonGrad"
        if "gTToonGrad" in shader_images:
            texture_node.image = shader_images["gTToonGrad"]

        uvs = create_node_group(nodes, "ToonGradUVs", toon_grad_uvs_node_group)
        links.new(uvs.outputs["Vector"], texture_node.inputs["Vector"])

        if normal_map is not None:
            links.new(normal_map.outputs["Normal"], uvs.inputs["Normal"])

        row_index = nodes.new("ShaderNodeValue")
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
        mix_ramp.data_type = "RGBA"
        mix_ramp.blend_type = "MULTIPLY"
        mix_ramp.inputs["Factor"].default_value = 1.0
        links.new(texture_node.outputs["Color"], mix_ramp.inputs["A"])
        links.new(mix_ao.outputs["Result"], mix_ramp.inputs["B"])

        if texture_node.image is not None:
            # Don't connect the toon nodes if the gradient texture is missing.
            # This avoids black material rendering.
            final_albedo = mix_ramp

    if material.state_flags.blend_mode == xc3_model_py.material.BlendMode.Multiply:
        # Workaround for Blender not supporting alpha blending modes.
        transparent_bsdf = nodes.new("ShaderNodeBsdfTransparent")
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

        node = nodes.get(name)
        if node is None:
            node = import_texture(name, nodes, textures)

        if channel == "Alpha":
            links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])
        else:
            rgb_node = nodes.new("ShaderNodeSeparateColor")
            links.new(node.outputs["Color"], rgb_node.inputs["Color"])
            links.new(rgb_node.outputs[channel], bsdf.inputs["Alpha"])

        # TODO: Support alpha blending?
        blender_material.blend_method = "CLIP"

    layout_nodes(output_node)

    return blender_material


def material_images_samplers(material, blender_images, samplers):
    material_textures = {}

    for i, texture in enumerate(material.textures):
        name = f"s{i}"

        image = None
        try:
            image = blender_images[texture.image_texture_index]
        except IndexError:
            pass

        # TODO: xenoblade x samplers?
        sampler = None
        if texture.sampler_index < len(samplers):
            sampler = samplers[texture.sampler_index]

        material_textures[name] = (image, sampler)

    return material_textures


def toon_grad_uvs_node_group():
    node_tree = bpy.data.node_groups.new("ToonGradUVs", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Row Index"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Normal"
    )

    uv = nodes.new("ShaderNodeCombineXYZ")

    # Using the actual lighting requires shader to RGB.
    # Use view based "lighting" to still support cycles.
    invert_facing = nodes.new("ShaderNodeMath")
    invert_facing.operation = "SUBTRACT"
    invert_facing.inputs[0].default_value = 1.0
    links.new(invert_facing.outputs["Value"], uv.inputs["X"])

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.inputs["Blend"].default_value = 0.5
    links.new(layer_weight.outputs["Facing"], invert_facing.inputs[1])
    links.new(input_node.outputs["Normal"], layer_weight.inputs["Normal"])

    flip_uvs = nodes.new("ShaderNodeMath")
    flip_uvs.label = "Flip UVs"
    flip_uvs.operation = "SUBTRACT"
    flip_uvs.inputs[0].default_value = 1.0
    links.new(flip_uvs.outputs["Value"], uv.inputs["Y"])

    div = nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 256.0
    links.new(div.outputs["Value"], flip_uvs.inputs[1])

    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.inputs[1].default_value = 0.5
    links.new(input_node.outputs["Row Index"], add.inputs[0])
    links.new(add.outputs["Value"], div.inputs[0])

    output_node = nodes.new("NodeGroupOutput")
    links.new(uv.outputs["Vector"], output_node.inputs["Vector"])

    layout_nodes(output_node)

    return node_tree


def create_node_group(nodes, name: str, create_node_tree):
    # Cache the node group creation.
    node_tree = bpy.data.node_groups.get(name)
    if node_tree is None:
        node_tree = create_node_tree()

    group = nodes.new("ShaderNodeGroup")
    group.node_tree = node_tree
    return group


def assign_normal_map(
    nodes,
    links,
    bsdf,
    x_assignment: int,
    y_assignment: int,
    intensity_assignment: Optional[int],
    assignments: list[xc3_model_py.material.Assignment],
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
):
    remap_normals = nodes.new("ShaderNodeVectorMath")
    remap_normals.operation = "MULTIPLY_ADD"
    remap_normals.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap_normals.inputs[2].default_value = (0.5, 0.5, 0.5)

    assign_normal_output(
        x_assignment,
        y_assignment,
        assignments,
        nodes,
        links,
        remap_normals.inputs[0],
        textures,
        is_data=True,
    )

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(remap_normals.outputs["Vector"], normal_map.inputs["Color"])

    if intensity_assignment is not None:
        assign_output(
            intensity_assignment,
            assignments,
            nodes,
            links,
            normal_map.inputs["Strength"],
            textures,
            is_data=True,
        )

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
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )

    remap_x = nodes.new("ShaderNodeMath")
    remap_x.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["X"], remap_x.inputs[0])
    remap_x.inputs[1].default_value = 2.0
    remap_x.inputs[2].default_value = -1.0

    remap_y = nodes.new("ShaderNodeMath")
    remap_y.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["Y"], remap_y.inputs[0])
    remap_y.inputs[1].default_value = 2.0
    remap_y.inputs[2].default_value = -1.0

    normal_xy = nodes.new("ShaderNodeCombineXYZ")
    links.new(remap_x.outputs["Value"], normal_xy.inputs["X"])
    links.new(remap_y.outputs["Value"], normal_xy.inputs["Y"])
    normal_xy.inputs["Z"].default_value = 0.0

    length2 = nodes.new("ShaderNodeVectorMath")
    length2.operation = "DOT_PRODUCT"
    links.new(normal_xy.outputs["Vector"], length2.inputs[0])
    links.new(normal_xy.outputs["Vector"], length2.inputs[1])

    one_minus_length = nodes.new("ShaderNodeMath")
    one_minus_length.operation = "SUBTRACT"
    one_minus_length.inputs[0].default_value = 1.0
    links.new(length2.outputs["Value"], one_minus_length.inputs[1])

    length = nodes.new("ShaderNodeMath")
    length.operation = "SQRT"
    links.new(one_minus_length.outputs["Value"], length.inputs[0])

    normal_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(remap_x.outputs["Value"], normal_xyz.inputs["X"])
    links.new(remap_y.outputs["Value"], normal_xyz.inputs["Y"])
    links.new(length.outputs["Value"], normal_xyz.inputs["Z"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(normal_xyz.outputs["Vector"], output_node.inputs["Normal"])

    layout_nodes(output_node)

    return node_tree


def add_normals_node_group():
    node_tree = bpy.data.node_groups.new("AddNormals", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Normal"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
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
    t.operation = "ADD"
    t.inputs[1].default_value = (0.0, 0.0, 1.0)
    links.new(input_node.outputs["A"], t.inputs[0])

    u = nodes.new("ShaderNodeVectorMath")
    u.operation = "MULTIPLY"
    u.inputs[1].default_value = (-1.0, -1.0, 1.0)
    links.new(input_node.outputs["B"], u.inputs[0])

    dot_t_u = nodes.new("ShaderNodeVectorMath")
    dot_t_u.operation = "DOT_PRODUCT"
    links.new(t.outputs["Vector"], dot_t_u.inputs[0])
    links.new(u.outputs["Vector"], dot_t_u.inputs[1])

    t_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(t.outputs["Vector"], t_xyz.inputs["Vector"])

    multiply_t = nodes.new("ShaderNodeVectorMath")
    multiply_t.operation = "MULTIPLY"
    links.new(dot_t_u.outputs["Value"], multiply_t.inputs[0])
    links.new(t.outputs["Vector"], multiply_t.inputs[1])

    multiply_u = nodes.new("ShaderNodeVectorMath")
    multiply_u.operation = "MULTIPLY"
    links.new(u.outputs["Vector"], multiply_u.inputs[0])
    links.new(t_xyz.outputs["Z"], multiply_u.inputs[1])

    r = nodes.new("ShaderNodeVectorMath")
    r.operation = "SUBTRACT"
    links.new(multiply_t.outputs["Vector"], r.inputs[0])
    links.new(multiply_u.outputs["Vector"], r.inputs[1])

    normalize_r = nodes.new("ShaderNodeVectorMath")
    normalize_r.operation = "NORMALIZE"
    links.new(r.outputs["Vector"], normalize_r.inputs[0])

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "VECTOR"
    links.new(input_node.outputs["Factor"], mix.inputs["Factor"])
    links.new(input_node.outputs["A"], mix.inputs["A"])
    links.new(normalize_r.outputs["Vector"], mix.inputs["B"])

    normalize_result = nodes.new("ShaderNodeVectorMath")
    normalize_result.operation = "NORMALIZE"
    links.new(mix.outputs["Result"], normalize_result.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(normalize_result.outputs["Vector"], output_node.inputs["Normal"])

    layout_nodes(output_node)

    return node_tree


def fresnel_blend_node_group():
    node_tree = bpy.data.node_groups.new("FresnelBlend", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Value"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Normal"
    )

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    links.new(input_node.outputs["Normal"], layer_weight.inputs["Normal"])

    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(input_node.outputs["Value"], multiply.inputs[0])
    multiply.inputs[1].default_value = 5.0

    pow_5 = nodes.new("ShaderNodeMath")
    pow_5.operation = "POWER"
    links.new(layer_weight.outputs["Facing"], pow_5.inputs[0])
    links.new(multiply.outputs["Value"], pow_5.inputs[1])

    output_node = nodes.new("NodeGroupOutput")
    links.new(pow_5.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node)

    return node_tree


def assign_output(
    assignment_index: int,
    assignments: list[xc3_model_py.material.Assignment],
    nodes,
    links,
    output,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
    is_data: bool,
):
    # Cache node creation to avoid creating too many nodes.
    # These names are unique to this material node tree.
    name = f"Assignment[{assignment_index}]"
    if node := nodes.get(name):
        links.new(node.outputs[0], output)
        return

    # Assign one output channel.
    if assignment_index >= len(assignments):
        return
    assignment = assignments[assignment_index]
    value = assignment.value()
    func = assignment.func()

    mix_rgba_node = lambda ty: assign_mix_rgba(
        func,
        assignments,
        nodes,
        links,
        output,
        textures,
        is_data,
        ty,
    )

    math_node = lambda ty: assign_math(
        func,
        assignments,
        nodes,
        links,
        output,
        textures,
        is_data,
        ty,
    )

    assign_index = lambda i, output: assign_output(
        i,
        assignments,
        nodes,
        links,
        output,
        textures,
        is_data,
    )

    if func is not None:
        match func.op:
            case xc3_model_py.shader_database.Operation.Mix:
                node = mix_rgba_node("MIX")
                node.name = name
            case xc3_model_py.shader_database.Operation.Mul:
                node = math_node("MULTIPLY")
                node.name = name
            case xc3_model_py.shader_database.Operation.Div:
                node = math_node("DIVIDE")
                node.name = name
            case xc3_model_py.shader_database.Operation.Add:
                node = math_node("ADD")
                node.name = name
            case xc3_model_py.shader_database.Operation.Sub:
                node = math_node("SUBTRACT")
                node.name = name
            case xc3_model_py.shader_database.Operation.Fma:
                node = math_node("MULTIPLY_ADD")
                node.name = name
            case xc3_model_py.shader_database.Operation.MulRatio:
                mix_values = nodes.new("ShaderNodeMix")
                mix_values.data_type = "RGBA"
                mix_values.blend_type = "MULTIPLY"

                mix_values.name = name

                links.new(mix_values.outputs["Result"], output)
                assign_index(func.args[0], mix_values.inputs["A"])
                assign_index(func.args[1], mix_values.inputs["B"])
                assign_index(func.args[2], mix_values.inputs["Factor"])
            case xc3_model_py.shader_database.Operation.AddNormal:
                mix_values = create_node_group(
                    nodes, "AddNormals", add_normals_node_group
                )
                mix_values.name = name
            case xc3_model_py.shader_database.Operation.Overlay:
                node = mix_rgba_node("OVERLAY")
                node.name = name
            case xc3_model_py.shader_database.Operation.Overlay2:
                node = mix_rgba_node("OVERLAY")
                node.name = name
            case xc3_model_py.shader_database.Operation.OverlayRatio:
                node = mix_rgba_node("OVERLAY")
            case xc3_model_py.shader_database.Operation.Power:
                node = math_node("POWER")
            case xc3_model_py.shader_database.Operation.Min:
                node = math_node("MINIMUM")
                node.name = name
            case xc3_model_py.shader_database.Operation.Max:
                node = math_node("MAXIMUM")
                node.name = name
            case xc3_model_py.shader_database.Operation.Clamp:
                pass
            case xc3_model_py.shader_database.Operation.Abs:
                node = math_node("ABSOLUTE")
                node.name = name
            case xc3_model_py.shader_database.Operation.Fresnel:
                node = create_node_group(
                    nodes, "FresnelBlend", fresnel_blend_node_group
                )
                node.name = name
                # TODO: normals?

                assign_index(func.args[0], node.inputs["Value"])
                links.new(node.outputs["Value"], output)
            case _:
                # TODO: This case shouldn't happen?
                pass
    elif value is not None:
        assign_value(value, nodes, links, output, textures, is_data)


def assign_value(
    value: xc3_model_py.material.AssignmentValue,
    nodes,
    links,
    output,
    textures,
    is_data,
):
    texture = value.texture()
    f = value.float()
    attribute = value.attribute()

    if f is not None:
        # This may be a single value or RGBA socket.
        try:
            output.default_value = f
        except:
            output.default_value = (f, f, f, 1.0)
    elif attribute is not None:
        assign_attribute(attribute, nodes, links, output)
    elif texture is not None:
        assign_texture(texture, nodes, links, output, textures, is_data)


def assign_attribute(attribute, nodes, links, output):
    node = nodes.get(attribute.name)
    if node is None:
        node = nodes.new("ShaderNodeAttribute")
        node.name = attribute.name

        if attribute.name == "vPos":
            node.attribute_name = "position"
        elif attribute.name == "vNormal":
            node.attribute_name = "VertexNormal"
        elif attribute.name == "vColor":
            node.attribute_name = "VertexColor"
        elif attribute.name == "vBlend":
            node.attribute_name = "Blend"
        else:
            for i in range(9):
                if attribute.name == f"vTex{i}":
                    node.attribute_name = f"TexCoord{i}"
                    break

        channel = channel_name(attribute.channel)
        if channel == "Alpha":
            links.new(node.outputs["Alpha"], output)
        else:
            # Avoid creating more than one separate RGB for each node.
            rgb_name = f"{attribute.name}.rgb"
            rgb_node = nodes.get(rgb_name)
            if rgb_node is None:
                rgb_node = nodes.new("ShaderNodeSeparateColor")
                rgb_node.name = rgb_name

            links.new(node.outputs["Color"], rgb_node.inputs["Color"])
            links.new(rgb_node.outputs[channel], output)


def assign_normal_output(
    assignment_index_x: int,
    assignment_index_y: int,
    assignments: list[xc3_model_py.material.Assignment],
    nodes,
    links,
    output,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
    is_data: bool,
):
    # Cache node creation to avoid creating too many nodes.
    # These names are unique to this material node tree.
    name = f"Assignment[{assignment_index_x}]"
    if node := nodes.get(name):
        links.new(node.outputs[0], output)
        return

    # Assign two output channels.
    if assignment_index_x >= len(assignments) or assignment_index_y >= len(assignments):
        return

    assignment_x = assignments[assignment_index_x]
    value_x = assignment_x.value()
    func_x = assignment_x.func()

    assignment_y = assignments[assignment_index_y]
    value_y = assignment_y.value()
    func_y = assignment_y.func()

    math_node = lambda ty: assign_normal_math(
        func_x,
        func_y,
        assignments,
        nodes,
        links,
        output,
        textures,
        is_data,
        ty,
    )

    # Assume x and y exprs have the same operations just with different arguments.
    if func_x is not None and func_y is not None:
        match func_x.op:
            case xc3_model_py.shader_database.Operation.Mix:
                node = nodes.new("ShaderNodeMix")
                node.data_type = "VECTOR"
                node.name = name

                assign_normal_output(
                    func_x.args[0],
                    func_y.args[0],
                    assignments,
                    nodes,
                    links,
                    node.inputs["A"],
                    textures,
                    is_data,
                )
                assign_normal_output(
                    func_x.args[1],
                    func_y.args[1],
                    assignments,
                    nodes,
                    links,
                    node.inputs["B"],
                    textures,
                    is_data,
                )

                # Assume the ratio is the same for x and y.
                assign_output(
                    func_x.args[2],
                    assignments,
                    nodes,
                    links,
                    node.inputs["Factor"],
                    textures,
                    is_data,
                )
            case xc3_model_py.shader_database.Operation.Mul:
                node = math_node("MULTIPLY")
                node.name = name
            case xc3_model_py.shader_database.Operation.Div:
                node = math_node("DIVIDE")
                node.name = name
            case xc3_model_py.shader_database.Operation.Add:
                node = math_node("ADD")
                node.name = name
            case xc3_model_py.shader_database.Operation.Sub:
                node = math_node("SUBTRACT")
                node.name = name
            case xc3_model_py.shader_database.Operation.Fma:
                node = math_node("MULTIPLY_ADD")
                node.name = name
            case xc3_model_py.shader_database.Operation.MulRatio:
                # TODO: How to handle this op with vectors?
                pass
            case xc3_model_py.shader_database.Operation.AddNormal:
                node = create_node_group(nodes, "AddNormals", add_normals_node_group)
                node.name = name

                assign_normal_output(
                    func_x.args[0],
                    func_y.args[0],
                    assignments,
                    nodes,
                    links,
                    node.inputs["A"],
                    textures,
                    is_data,
                )
                assign_normal_output(
                    func_x.args[1],
                    func_y.args[1],
                    assignments,
                    nodes,
                    links,
                    node.inputs["B"],
                    textures,
                    is_data,
                )

                # Assume the ratio is the same for x and y.
                assign_output(
                    func_x.args[2],
                    assignments,
                    nodes,
                    links,
                    node.inputs["Factor"],
                    textures,
                    is_data,
                )

                links.new(node.outputs["Normal"], output)

            case xc3_model_py.shader_database.Operation.Overlay:
                # TODO: How to handle this op with vectors?
                pass
            case xc3_model_py.shader_database.Operation.Overlay2:
                # TODO: How to handle this op with vectors?
                pass
            case xc3_model_py.shader_database.Operation.OverlayRatio:
                # TODO: How to handle this op with vectors?
                pass
            case xc3_model_py.shader_database.Operation.Power:
                node = math_node("POWER")
                node.name = name
            case xc3_model_py.shader_database.Operation.Min:
                node = math_node("MINIMUM")
                node.name = name
            case xc3_model_py.shader_database.Operation.Max:
                node = math_node("MAXIMUM")
                node.name = name
            case xc3_model_py.shader_database.Operation.Clamp:
                pass
            case xc3_model_py.shader_database.Operation.Abs:
                node = math_node("ABSOLUTE")
                node.name = name
            case xc3_model_py.shader_database.Operation.Fresnel:
                # TODO: How to handle this op with vectors?
                pass
            case _:
                mix_values = nodes.new("ShaderNodeMix")
                mix_values.data_type = "RGBA"

                mix_values.name = name
    elif value_x is not None and value_y is not None:
        node = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
        node.name = name

        assign_value(
            value_x,
            nodes,
            links,
            node.inputs["X"],
            textures,
            is_data,
        )
        assign_value(
            value_y,
            nodes,
            links,
            node.inputs["Y"],
            textures,
            is_data,
        )

        links.new(node.outputs["Normal"], output)


def assign_mix_rgba(
    func,
    assignments: list[xc3_model_py.material.Assignment],
    nodes,
    links,
    output,
    textures,
    is_data,
    blend_type: str,
):
    mix_values = nodes.new("ShaderNodeMix")
    mix_values.data_type = "RGBA"
    mix_values.blend_type = blend_type
    mix_values.inputs["Factor"].default_value = 1.0

    links.new(mix_values.outputs["Result"], output)

    assign_output(
        func.args[0],
        assignments,
        nodes,
        links,
        mix_values.inputs["A"],
        textures,
        is_data,
    )
    assign_output(
        func.args[1],
        assignments,
        nodes,
        links,
        mix_values.inputs["B"],
        textures,
        is_data,
    )
    if len(func.args) == 3:
        assign_output(
            func.args[2],
            assignments,
            nodes,
            links,
            mix_values.inputs["Factor"],
            textures,
            is_data,
        )

    return mix_values


def assign_texture(
    texture: xc3_model_py.material.TextureAssignment,
    nodes,
    links,
    output,
    textures,
    is_data,
):
    # Load only the textures that are actually used.
    node = nodes.get(texture.name)
    if node is None:
        node = import_texture(texture.name, nodes, textures)

    # TODO: Find a better way to handle color management.
    # TODO: Why can't we just set everything to non color?
    # TODO: This won't work if users have different color spaces installed like aces.
    if node.image is not None:
        if is_data:
            node.image.colorspace_settings.name = "Non-Color"
        else:
            node.image.colorspace_settings.name = "sRGB"

    channel = channel_name(texture.channel)
    if channel == "Alpha":
        # Alpha isn't part of the RGB node.
        links.new(node.outputs["Alpha"], output)
    else:
        # Avoid creating more than one separate RGB for each texture.
        rgb_name = f"{texture.name}.rgb"
        rgb_node = nodes.get(rgb_name)
        if rgb_node is None:
            rgb_node = nodes.new("ShaderNodeSeparateColor")
            rgb_node.name = rgb_name

        links.new(node.outputs["Color"], rgb_node.inputs["Color"])
        links.new(rgb_node.outputs[channel], output)

    uv_name = texture.texcoord_name or ""
    uv = nodes.get(uv_name)
    if uv is None:
        uv = nodes.new("ShaderNodeUVMap")
        uv.name = uv_name
        for i in range(9):
            if texture.texcoord_name == f"vTex{i}":
                uv.uv_map = f"TexCoord{i}"
                break

    # TODO: Create a node group for the mat2x4 transform (two dot products).
    if texture.texcoord_transforms is not None:
        transform_u, transform_v = texture.texcoord_transforms
        name = f"{uv_name}_{transform_u}_{transform_v}"
        scale = nodes.get(name)
        if scale is None:
            # TODO: Use the full mat2x4 transform.
            scale = nodes.new("ShaderNodeVectorMath")
            scale.name = name
            scale.operation = "MULTIPLY"
            scale.inputs[1].default_value = (
                transform_u[0],
                transform_v[1],
                1.0,
            )
            links.new(uv.outputs["UV"], scale.inputs["Vector"])

        links.new(scale.outputs["Vector"], node.inputs["Vector"])
    else:
        links.new(uv.outputs["UV"], node.inputs["Vector"])

    return node


def import_texture(
    name: str,
    nodes,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
):
    node = nodes.new("ShaderNodeTexImage")
    node.name = name
    node.label = name

    if name in textures:
        image, sampler = textures[name]
        node.image = image

        if sampler is not None:
            # TODO: Check if U and V have the same address mode.
            match sampler.address_mode_u:
                case xc3_model_py.AddressMode.ClampToEdge:
                    node.extension = "CLIP"
                case xc3_model_py.AddressMode.Repeat:
                    node.extension = "REPEAT"
                case xc3_model_py.AddressMode.MirrorRepeat:
                    node.extension = "MIRROR"

    return node


def assign_math(
    func,
    assignments: list[xc3_model_py.material.Assignment],
    nodes,
    links,
    output,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
    is_data: bool,
    op: str,
) -> bpy.types.Node:
    node = nodes.new("ShaderNodeMath")
    node.operation = op

    links.new(node.outputs["Value"], output)
    for arg, input in zip(func.args, node.inputs):
        assign_output(
            arg,
            assignments,
            nodes,
            links,
            input,
            textures,
            is_data,
        )

    return node


def assign_normal_math(
    func_x,
    func_y,
    assignments: list[xc3_model_py.material.Assignment],
    nodes,
    links,
    output,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
    is_data: bool,
    op: str,
) -> bpy.types.Node:
    node = nodes.new("ShaderNodeMath")
    node.operation = op

    links.new(node.outputs["Value"], output)
    for arg_x, arg_y, input in zip(func_x.args, func_y.args, node.inputs):
        assign_normal_output(
            arg_x,
            arg_y,
            assignments,
            nodes,
            links,
            input,
            textures,
            is_data,
        )

    return node


def channel_name(channel: Optional[str]) -> str:
    match channel:
        case "x":
            return "Red"
        case "y":
            return "Green"
        case "z":
            return "Blue"
        case "w":
            return "Alpha"

    return ""
