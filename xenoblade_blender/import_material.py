from typing import Dict, Optional, Tuple
import bpy
import typing

from xenoblade_blender.node_group import (
    add_normals_node_group,
    create_node_group,
    fresnel_blend_node_group,
    normal_map_xy_final_node_group,
    normal_map_xyz_node_group,
    reflect_xyz_node_group,
    tex_matrix_node_group,
    tex_parallax_node_group,
    toon_grad_uvs_node_group,
)
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
    normals = create_node_group(
        nodes, "NormalMapXYFinal", normal_map_xy_final_node_group
    )
    normals.inputs["X"].default_value = 0.5
    normals.inputs["Y"].default_value = 0.5
    normals.inputs["Strength"].default_value = 1.0

    assign_output(
        x_assignment,
        assignments,
        nodes,
        links,
        normals.inputs["X"],
        textures,
        is_data=True,
    )
    assign_output(
        y_assignment,
        assignments,
        nodes,
        links,
        normals.inputs["Y"],
        textures,
        is_data=True,
    )

    if intensity_assignment is not None:
        assign_output(
            intensity_assignment,
            assignments,
            nodes,
            links,
            normals.inputs["Strength"],
            textures,
            is_data=True,
        )

    links.new(normals.outputs["Normal"], bsdf.inputs["Normal"])

    return normals


def assign_output(
    assignment_index: Optional[int],
    assignments: list[xc3_model_py.material.Assignment],
    nodes,
    links,
    output,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
    is_data: bool,
):
    if assignment_index is None:
        return

    if assignment_index >= len(assignments):
        return

    # Cache node creation to avoid creating too many nodes.
    # These names are unique to this material node tree.
    name = f"Assignment[{assignment_index}]"
    if node := nodes.get(name):
        links.new(node.outputs[0], output)
        return

    # Assign one output channel.
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
            case xc3_model_py.shader_database.Operation.Unk:
                # Set defaults to match xc3_wgpu and make debugging easier.
                assign_float(output, 0.0)
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
            case xc3_model_py.shader_database.Operation.AddNormalX:
                # TODO: Share with Y based on the input indices?
                mix_values = create_node_group(
                    nodes, "AddNormals", add_normals_node_group
                )
                mix_values.name = name

                links.new(mix_values.outputs["X"], output)
                assign_index(func.args[0], mix_values.inputs["A.x"])
                assign_index(func.args[1], mix_values.inputs["A.y"])
                assign_index(func.args[2], mix_values.inputs["B.x"])
                assign_index(func.args[3], mix_values.inputs["B.y"])
                assign_index(func.args[4], mix_values.inputs["Factor"])
            case xc3_model_py.shader_database.Operation.AddNormalY:
                # TODO: Share with X based on the input indices?
                mix_values = create_node_group(
                    nodes, "AddNormals", add_normals_node_group
                )
                mix_values.name = name

                links.new(mix_values.outputs["Y"], output)
                assign_index(func.args[0], mix_values.inputs["A.x"])
                assign_index(func.args[1], mix_values.inputs["A.y"])
                assign_index(func.args[2], mix_values.inputs["B.x"])
                assign_index(func.args[3], mix_values.inputs["B.y"])
                assign_index(func.args[4], mix_values.inputs["Factor"])
            case xc3_model_py.shader_database.Operation.Overlay:
                node = mix_rgba_node("OVERLAY")
                node.inputs["Factor"].default_value = 1.0
                node.name = name
            case xc3_model_py.shader_database.Operation.Overlay2:
                node = mix_rgba_node("OVERLAY")
                node.inputs["Factor"].default_value = 1.0
                node.name = name
            case xc3_model_py.shader_database.Operation.OverlayRatio:
                node = mix_rgba_node("OVERLAY")
                node.name = name
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
                node = nodes.new("ShaderNodeClamp")
                node.name = name

                links.new(node.outputs["Result"], output)
                assign_index(func.args[0], node.inputs["Value"])
                assign_index(func.args[1], node.inputs["Min"])
                assign_index(func.args[2], node.inputs["Max"])
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
            case xc3_model_py.shader_database.Operation.Sqrt:
                node = math_node("SQRT")
                node.name = name
            case xc3_model_py.shader_database.Operation.TexMatrix:
                node = create_node_group(nodes, "TexMatrix", tex_matrix_node_group)
                node.name = name

                links.new(node.outputs["Value"], output)
                assign_index(func.args[0], node.inputs["U"])
                assign_index(func.args[1], node.inputs["V"])
                assign_index(func.args[2], node.inputs["A"])
                assign_index(func.args[3], node.inputs["B"])
                assign_index(func.args[4], node.inputs["C"])
                assign_index(func.args[5], node.inputs["D"])
            case xc3_model_py.shader_database.Operation.TexParallaxX:
                # TODO: Separate node groups for each channel or split XY?
                node = create_node_group(nodes, "TexParallax", tex_parallax_node_group)
                node.name = name

                links.new(node.outputs["Value"], output)
                assign_index(func.args[0], node.inputs["Value"])
                assign_index(func.args[1], node.inputs["Factor"])
            case xc3_model_py.shader_database.Operation.TexParallaxY:
                node = create_node_group(nodes, "TexParallax", tex_parallax_node_group)
                node.name = name

                links.new(node.outputs["Value"], output)
                assign_index(func.args[0], node.inputs["Value"])
                assign_index(func.args[1], node.inputs["Factor"])
            case xc3_model_py.shader_database.Operation.ReflectX:
                node = create_node_group(nodes, "ReflectXYZ", reflect_xyz_node_group)
                node.name = name

                links.new(node.outputs["X"], output)
                assign_index(func.args[0], node.inputs["A.x"])
                assign_index(func.args[1], node.inputs["A.y"])
                assign_index(func.args[2], node.inputs["A.z"])
                assign_index(func.args[3], node.inputs["B.x"])
                assign_index(func.args[4], node.inputs["B.y"])
                assign_index(func.args[5], node.inputs["B.z"])
            case xc3_model_py.shader_database.Operation.ReflectY:
                # TODO: Share with X based on the input indices?
                node = create_node_group(nodes, "ReflectXYZ", reflect_xyz_node_group)
                node.name = name

                links.new(node.outputs["Y"], output)
                assign_index(func.args[0], node.inputs["A.x"])
                assign_index(func.args[1], node.inputs["A.y"])
                assign_index(func.args[2], node.inputs["A.z"])
                assign_index(func.args[3], node.inputs["B.x"])
                assign_index(func.args[4], node.inputs["B.y"])
                assign_index(func.args[5], node.inputs["B.z"])
            case xc3_model_py.shader_database.Operation.ReflectZ:
                # TODO: Share with X based on the input indices?
                node = create_node_group(nodes, "ReflectXYZ", reflect_xyz_node_group)
                node.name = name

                links.new(node.outputs["Z"], output)
                assign_index(func.args[0], node.inputs["A.x"])
                assign_index(func.args[1], node.inputs["A.y"])
                assign_index(func.args[2], node.inputs["A.z"])
                assign_index(func.args[3], node.inputs["B.x"])
                assign_index(func.args[4], node.inputs["B.y"])
                assign_index(func.args[5], node.inputs["B.z"])
            case xc3_model_py.shader_database.Operation.Floor:
                node = math_node("FLOOR")
                node.name = name
            case xc3_model_py.shader_database.Operation.Select:
                node = mix_rgba_node("MIX")
                node.name = name
            case xc3_model_py.shader_database.Operation.Equal:
                node = math_node("COMPARE")
                node.name = name
            case xc3_model_py.shader_database.Operation.NotEqual:
                # TODO: Invert compare.
                pass
            case xc3_model_py.shader_database.Operation.Less:
                node = math_node("LESS_THAN")
                node.name = name
            case xc3_model_py.shader_database.Operation.Greater:
                node = math_node("GREATER_THAN")
                node.name = name
            case xc3_model_py.shader_database.Operation.LessEqual:
                # TODO: node group?
                node = math_node("LESS_THAN")
                node.name = name
            case xc3_model_py.shader_database.Operation.GreaterEqual:
                # TODO: node group?
                node = math_node("GREATER_THAN")
                node.name = name
            case xc3_model_py.shader_database.Operation.Dot4:
                pass
            case xc3_model_py.shader_database.Operation.NormalMapX:
                node = create_node_group(
                    nodes, "NormalMapXYZ", normal_map_xyz_node_group
                )
                node.name = name

                links.new(node.outputs["X"], output)
                assign_index(func.args[0], node.inputs["X"])
                assign_index(func.args[1], node.inputs["Y"])
            case xc3_model_py.shader_database.Operation.NormalMapY:
                # TODO: Share with X based on the input indices?
                node = create_node_group(
                    nodes, "NormalMapXYZ", normal_map_xyz_node_group
                )
                node.name = name

                links.new(node.outputs["Y"], output)
                assign_index(func.args[0], node.inputs["X"])
                assign_index(func.args[1], node.inputs["Y"])
            case xc3_model_py.shader_database.Operation.NormalMapZ:
                # TODO: Share with X based on the input indices?
                node = create_node_group(
                    nodes, "NormalMapXYZ", normal_map_xyz_node_group
                )
                node.name = name

                links.new(node.outputs["Z"], output)
                assign_index(func.args[0], node.inputs["X"])
                assign_index(func.args[1], node.inputs["Y"])
            case _:
                # TODO: This case shouldn't happen?
                # Set defaults to match xc3_wgpu and make debugging easier.
                assign_float(output, 0.0)
    elif value is not None:
        assign_value(value, assignments, nodes, links, output, textures, is_data)


def assign_value(
    value: xc3_model_py.material.AssignmentValue,
    assignments,
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
        assign_float(output, f)
    elif attribute is not None:
        assign_attribute(attribute, nodes, links, output)
    elif texture is not None:
        assign_texture(texture, assignments, nodes, links, output, textures, is_data)


def assign_float(output, f):
    # This may be a float, RGBA, or XYZ socket.
    try:
        output.default_value = f
    except:
        output.default_value = [f] * len(output.default_value)


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

    links.new(mix_values.outputs["Result"], output)

    # Set defaults to match xc3_wgpu and make debugging easier.
    for input in mix_values.inputs:
        assign_float(input, 0.0)

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
    assignments,
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

    # Texture coordinates can be made of multiple nodes.
    uv_name = f"uv{texture.texcoords}"
    uv_node = nodes.get(uv_name)
    if uv_node is None:
        uv_node = nodes.new("ShaderNodeCombineXYZ")
        uv_node.name = uv_name

        if len(texture.texcoords) >= 2:
            assign_output(
                texture.texcoords[0],
                assignments,
                nodes,
                links,
                uv_node.inputs["X"],
                textures,
                is_data=True,
            )
            assign_output(
                texture.texcoords[1],
                assignments,
                nodes,
                links,
                uv_node.inputs["Y"],
                textures,
                is_data=True,
            )

    links.new(uv_node.outputs["Vector"], node.inputs["Vector"])

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

    # Set defaults to match xc3_wgpu and make debugging easier.
    for input in node.inputs:
        assign_float(input, 0.0)

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

    # TODO: How to handle the None case?
    return "Red"
