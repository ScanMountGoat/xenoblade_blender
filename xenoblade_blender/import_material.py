from typing import Dict, Optional, Set, Tuple
import bpy
import typing

from xenoblade_blender.node_group import (
    add_normals_node_group,
    clamp_xyz_node_group,
    create_node_group,
    fresnel_blend_node_group,
    greater_xyz_node_group,
    less_xyz_node_group,
    monochrome_xyz_node_group,
    normal_map_xy_final_node_group,
    normal_map_xyz_node_group,
    power_xyz_node_group,
    reflect_xyz_node_group,
    sqrt_xyz_node_group,
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
    material: xc3_model_py.material.Material,
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

    has_alpha = material.state_flags.blend_mode not in [
        xc3_model_py.material.BlendMode.Disabled,
        xc3_model_py.material.BlendMode.Disabled2,
    ]

    # Create nodes for each unique assignment.
    # Storing the output name allows using a single node for values with multiple channels.
    assignment_indices = used_assignments(output_assignments, has_alpha)
    assignment_outputs = []
    for i, assignment in enumerate(output_assignments.assignments):
        if i in assignment_indices:
            node_output = assign_output(
                assignment, assignment_outputs, nodes, links, textures
            )
            assignment_outputs.append(node_output)
        else:
            assignment_outputs.append(None)

    if has_alpha:
        assign_index(
            output_assignments.output_assignments[0].w,
            assignment_outputs,
            links,
            bsdf.inputs["Alpha"],
        )

    mix_ao = nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mix_ao.inputs["B"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix_ao.inputs["Factor"].default_value = 1.0

    # RGB base color.
    if xyz := output_assignments.output_assignments[0].merge_xyz(
        output_assignments.assignments
    ):
        assignment_outputs_xyz = create_assignment_outputs_xyz(
            xyz,
            assignment_outputs,
            nodes,
            links,
            textures,
        )

        assign_index(
            xyz.assignment,
            assignment_outputs_xyz,
            links,
            mix_ao.inputs["A"],
        )
    else:
        base_color = nodes.new("ShaderNodeCombineColor")
        links.new(base_color.outputs["Color"], mix_ao.inputs["A"])

        assign_index(
            output_assignments.output_assignments[0].x,
            assignment_outputs,
            links,
            base_color.inputs["Red"],
        )
        assign_index(
            output_assignments.output_assignments[0].y,
            assignment_outputs,
            links,
            base_color.inputs["Green"],
        )
        assign_index(
            output_assignments.output_assignments[0].z,
            assignment_outputs,
            links,
            base_color.inputs["Blue"],
        )

    # Single channel ambient occlusion.
    assign_index(
        output_assignments.output_assignments[2].z,
        assignment_outputs,
        links,
        mix_ao.inputs["B"],
    )

    normal_map = assign_normal_map(
        nodes,
        links,
        bsdf,
        output_assignments.output_assignments[2].x,
        output_assignments.output_assignments[2].y,
        output_assignments.normal_intensity,
        assignment_outputs,
    )

    assign_index(
        output_assignments.output_assignments[1].x,
        assignment_outputs,
        links,
        bsdf.inputs["Metallic"],
    )

    if (
        output_assignments.output_assignments[5].x is not None
        or output_assignments.output_assignments[5].y is not None
        or output_assignments.output_assignments[5].z is not None
    ):
        # TODO: Toon and hair shaders always use specular color?
        # Xenoblade X models typically use specular but don't have a mat id value yet.
        # TODO: use the material render flags instead for better accuracy.
        if mat_id in [2, 5] or mat_id is None:
            output = bsdf.inputs["Specular Tint"]
        else:
            output = bsdf.inputs["Emission Color"]
            bsdf.inputs["Emission Strength"].default_value = 1.0

        if xyz := output_assignments.output_assignments[5].merge_xyz(
            output_assignments.assignments
        ):
            assignment_outputs_xyz = create_assignment_outputs_xyz(
                xyz,
                assignment_outputs,
                nodes,
                links,
                textures,
            )

            assign_index(xyz.assignment, assignment_outputs_xyz, links, output)
        else:
            color = nodes.new("ShaderNodeCombineColor")
            assign_index(
                output_assignments.output_assignments[5].x,
                assignment_outputs,
                links,
                color.inputs["Red"],
            )
            assign_index(
                output_assignments.output_assignments[5].y,
                assignment_outputs,
                links,
                color.inputs["Green"],
            )
            assign_index(
                output_assignments.output_assignments[5].z,
                assignment_outputs,
                links,
                color.inputs["Blue"],
            )
            links.new(color.outputs["Color"], output)

    # Invert glossiness to get roughness.
    invert = nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0
    assign_index(
        output_assignments.output_assignments[1].y,
        assignment_outputs,
        links,
        invert.inputs[1],
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

    # In game textures use _UNORM formats and write to a non sRGB texture.
    # The deferred lighting shaders for all games use gamma 2.2 instead of sRGB.
    base_color = nodes.new("ShaderNodeGamma")
    links.new(final_albedo.outputs["Result"], base_color.inputs["Color"])
    base_color.inputs["Gamma"].default_value = 2.2

    if material.state_flags.blend_mode == xc3_model_py.material.BlendMode.Multiply:
        # Workaround for Blender not supporting alpha blending modes.
        transparent_bsdf = nodes.new("ShaderNodeBsdfTransparent")
        links.new(base_color.outputs["Color"], transparent_bsdf.inputs["Color"])
        links.new(transparent_bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    else:
        links.new(base_color.outputs["Color"], bsdf.inputs["Base Color"])

    if has_alpha:
        blender_material.blend_method = "BLEND"

    if material.alpha_test is not None:
        texture = material.alpha_test
        name = f"s{texture.texture_index}"
        channel = ["Red", "Green", "Blue", "Alpha"][texture.channel_index]

        node = nodes.get(name)
        if node is None:
            node = import_texture(name, name, nodes, textures)

        if channel == "Alpha":
            links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])
        else:
            rgb_node = nodes.new("ShaderNodeSeparateColor")
            links.new(node.outputs["Color"], rgb_node.inputs["Color"])
            links.new(rgb_node.outputs[channel], bsdf.inputs["Alpha"])

        # TODO: Support alpha blending?
        blender_material.blend_method = "CLIP"

    layout_nodes(output_node, links)

    return blender_material


def create_assignment_outputs_xyz(
    xyz: xc3_model_py.material.OutputAssignmentXyz,
    assignment_outputs,
    nodes,
    links,
    textures,
):
    # Create nodes for each unique assignment.
    # Storing the output name allows using a single node for values with multiple channels.
    assignment_outputs_xyz = []
    for assignment in xyz.assignments:
        node_output = assign_output_xyz(
            assignment,
            assignment_outputs,
            assignment_outputs_xyz,
            nodes,
            links,
            textures,
        )
        assignment_outputs_xyz.append(node_output)

    return assignment_outputs_xyz


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
    x_assignment: Optional[int],
    y_assignment: Optional[int],
    intensity_assignment: Optional[int],
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
) -> Optional[bpy.types.Node]:
    if x_assignment is None or y_assignment is None:
        return None

    normals = create_node_group(
        nodes, "NormalMapXYFinal", normal_map_xy_final_node_group
    )
    normals.inputs["X"].default_value = 0.5
    normals.inputs["Y"].default_value = 0.5
    normals.inputs["Strength"].default_value = 1.0

    assign_index(
        x_assignment,
        assignment_outputs,
        links,
        normals.inputs["X"],
    )
    assign_index(
        y_assignment,
        assignment_outputs,
        links,
        normals.inputs["Y"],
    )

    if intensity_assignment is not None:
        assign_index(
            intensity_assignment,
            assignment_outputs,
            links,
            normals.inputs["Strength"],
        )

    links.new(normals.outputs["Normal"], bsdf.inputs["Normal"])

    return normals


def assign_output(
    assignment: xc3_model_py.material.Assignment,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
) -> Optional[Tuple[bpy.types.Node, str]]:
    if func := assignment.func():
        return assign_func(func, assignment_outputs, nodes, links)
    elif value := assignment.value():
        return assign_value(value, assignment_outputs, nodes, links, textures)
    else:
        return None


def assign_func(
    func: xc3_model_py.material.AssignmentFunc,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str]]:
    mix_rgba_node = lambda ty: assign_mix_rgba(
        func,
        assignment_outputs,
        nodes,
        links,
        ty,
    )

    math_node = lambda ty: assign_math(
        func,
        assignment_outputs,
        nodes,
        links,
        ty,
    )

    group_node = lambda func, name, create_node_tree: create_cached_func_group_node(
        nodes, func, name, create_node_tree
    )

    assign_args = lambda func, node, params: assign_func_args(
        func, params, assignment_outputs, links, node
    )

    match func.op:
        case xc3_model_py.shader_database.Operation.Unk:
            return None
        case xc3_model_py.shader_database.Operation.Mix:
            return mix_rgba_node("MIX")
        case xc3_model_py.shader_database.Operation.Mul:
            return math_node("MULTIPLY")
        case xc3_model_py.shader_database.Operation.Div:
            return math_node("DIVIDE")
        case xc3_model_py.shader_database.Operation.Add:
            return math_node("ADD")
        case xc3_model_py.shader_database.Operation.Sub:
            return math_node("SUBTRACT")
        case xc3_model_py.shader_database.Operation.Fma:
            return math_node("MULTIPLY_ADD")
        case xc3_model_py.shader_database.Operation.MulRatio:
            node = nodes.new("ShaderNodeMix")
            node.data_type = "RGBA"
            node.blend_type = "MULTIPLY"
            node.name = func_name(func)
            assign_args(func, node, ["A", "B", "Factor"])
            return node, "Result"
        case xc3_model_py.shader_database.Operation.AddNormalX:
            # Reuse the node for other channels if possible.
            node = group_node(func, "AddNormals", add_normals_node_group)
            assign_args(func, node, ["A.x", "A.y", "B.x", "B.y", "Factor"])
            return node, "X"
        case xc3_model_py.shader_database.Operation.AddNormalY:
            # Reuse the node for other channels if possible.
            node = group_node(func, "AddNormals", add_normals_node_group)
            assign_args(func, node, ["A.x", "A.y", "B.x", "B.y", "Factor"])
            return node, "Y"
        case xc3_model_py.shader_database.Operation.Overlay:
            return mix_rgba_node("OVERLAY")
        case xc3_model_py.shader_database.Operation.Overlay2:
            return mix_rgba_node("OVERLAY")
        case xc3_model_py.shader_database.Operation.OverlayRatio:
            return mix_rgba_node("OVERLAY")
        case xc3_model_py.shader_database.Operation.Power:
            return math_node("POWER")
        case xc3_model_py.shader_database.Operation.Min:
            return math_node("MINIMUM")
        case xc3_model_py.shader_database.Operation.Max:
            return math_node("MAXIMUM")
        case xc3_model_py.shader_database.Operation.Clamp:
            node = nodes.new("ShaderNodeClamp")
            node.name = func_name(func)
            assign_args(func, node, ["Value", "Min", "Max"])
            return node, "Result"
        case xc3_model_py.shader_database.Operation.Abs:
            return math_node("ABSOLUTE")
        case xc3_model_py.shader_database.Operation.Fresnel:
            node = group_node(func, "FresnelBlend", fresnel_blend_node_group)
            # TODO: normals?
            assign_args(func, node, ["Value"])
            return node, "Value"
        case xc3_model_py.shader_database.Operation.Sqrt:
            return math_node("SQRT")
        case xc3_model_py.shader_database.Operation.TexMatrix:
            node = group_node(func, "TexMatrix", tex_matrix_node_group)
            assign_args(func, node, ["U", "V", "A", "B", "C", "D"])
            return node, "Value"
        case xc3_model_py.shader_database.Operation.TexParallaxX:
            node = group_node(func, "TexParallax", tex_parallax_node_group)
            assign_args(func, node, ["Value", "Factor"])
            return node, "Value"
        case xc3_model_py.shader_database.Operation.TexParallaxY:
            node = group_node(func, "TexParallax", tex_parallax_node_group)
            assign_args(func, node, ["Value", "Factor"])
            return node, "Value"
        case xc3_model_py.shader_database.Operation.ReflectX:
            # Reuse the node for other channels if possible.
            node = group_node(func, "ReflectXYZ", reflect_xyz_node_group)
            assign_args(func, node, ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"])
            return node, "X"
        case xc3_model_py.shader_database.Operation.ReflectY:
            # Reuse the node for other channels if possible.
            node = group_node(func, "ReflectXYZ", reflect_xyz_node_group)
            assign_args(func, node, ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"])
            return node, "Y"
        case xc3_model_py.shader_database.Operation.ReflectZ:
            # Reuse the node for other channels if possible.
            node = group_node(func, "ReflectXYZ", reflect_xyz_node_group)
            assign_args(func, node, ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"])
            return node, "Z"
        case xc3_model_py.shader_database.Operation.Floor:
            return math_node("FLOOR")
        case xc3_model_py.shader_database.Operation.Select:
            return mix_rgba_node("MIX")
        case xc3_model_py.shader_database.Operation.Equal:
            return math_node("COMPARE")
        case xc3_model_py.shader_database.Operation.NotEqual:
            # TODO: Invert compare.
            return math_node("COMPARE")
        case xc3_model_py.shader_database.Operation.Less:
            return math_node("LESS_THAN")
        case xc3_model_py.shader_database.Operation.Greater:
            return math_node("GREATER_THAN")
        case xc3_model_py.shader_database.Operation.LessEqual:
            # TODO: node group for leq?
            return math_node("LESS_THAN")
        case xc3_model_py.shader_database.Operation.GreaterEqual:
            # TODO: node group for geq?
            return math_node("GREATER_THAN")
        case xc3_model_py.shader_database.Operation.Dot4:
            pass
        case xc3_model_py.shader_database.Operation.NormalMapX:
            # Reuse the node for other channels if possible.
            node = group_node(func, "NormalMapXYZ", normal_map_xyz_node_group)
            assign_args(func, node, ["X", "Y"])
            return node, "X"
        case xc3_model_py.shader_database.Operation.NormalMapY:
            # Reuse the node for other channels if possible.
            node = group_node(func, "NormalMapXYZ", normal_map_xyz_node_group)
            assign_args(func, node, ["X", "Y"])
            return node, "Y"
        case xc3_model_py.shader_database.Operation.NormalMapZ:
            # Reuse the node for other channels if possible.
            node = group_node(func, "NormalMapXYZ", normal_map_xyz_node_group)
            assign_args(func, node, ["X", "Y"])
            return node, "Z"
        case xc3_model_py.shader_database.Operation.MonochromeX:
            # Reuse the node for other channels if possible.
            node = group_node(func, "MonochromeXYZ", monochrome_xyz_node_group)
            assign_args(func, node, ["X", "Y", "Z", "Factor"])
            return node, "X"
        case xc3_model_py.shader_database.Operation.MonochromeY:
            # Reuse the node for other channels if possible.
            node = group_node(func, "MonochromeXYZ", monochrome_xyz_node_group)
            assign_args(func, node, ["X", "Y", "Z", "Factor"])
            return node, "Y"
        case xc3_model_py.shader_database.Operation.MonochromeZ:
            # Reuse the node for other channels if possible.
            node = group_node(func, "MonochromeXYZ", monochrome_xyz_node_group)
            assign_args(func, node, ["X", "Y", "Z", "Factor"])
            return node, "Z"
        case xc3_model_py.shader_database.Operation.Negate:
            node = nodes.new("ShaderNodeMath")
            node.name = func_name(func)
            node.operation = "MULTIPLY"

            assign_args(func, node, [0])
            node.inputs[1].default_value = -1.0

            return node, "Value"
        case xc3_model_py.shader_database.Operation.FurInstanceAlpha:
            node = nodes.new("ShaderNodeAttribute")
            node.name = func_name(func)
            node.attribute_name = "FurAlpha"
            return node, "Fac"
        case _:
            # TODO: This case shouldn't happen?
            return None


def assign_index(
    i: Optional[int],
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    links,
    output,
):
    if i is not None:
        if node_output := assignment_outputs[i]:
            node, output_name = node_output
            links.new(node.outputs[output_name], output)
        else:
            # Set defaults to match xc3_wgpu and make debugging easier.
            assign_float(output, 0.0)


def assign_value(
    value: xc3_model_py.material.AssignmentValue,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures,
) -> Optional[Tuple[bpy.types.Node, str]]:
    if f := value.float():
        node = nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = f
        return node, "Value"
    elif attribute := value.attribute():
        return assign_attribute(attribute, nodes, links)
    elif texture := value.texture():
        return assign_texture(texture, assignment_outputs, nodes, links, textures)
    else:
        return None


def assign_float(output, f):
    # This may be a float, RGBA, or XYZ socket.
    try:
        output.default_value = [f] * len(output.default_value)
    except:
        output.default_value = f


def assign_attribute(
    attribute: xc3_model_py.material.AssignmentValueAttribute, nodes, links
) -> Tuple[bpy.types.Node, str]:
    node = import_attribute(attribute.name, nodes)
    return assign_channel(attribute.name, attribute.channel, node, nodes, links)


def assign_channel(
    name: str, channel: Optional[str], node, nodes, links
) -> Tuple[bpy.types.Node, str]:
    output = channel_name(channel)
    if output == "Alpha":
        # Alpha isn't part of the RGB node.
        return node, "Alpha"
    else:
        # Avoid creating more than one separate RGB for each node.
        rgb_name = f"{name}.rgb"
        rgb_node = nodes.get(rgb_name)
        if rgb_node is None:
            rgb_node = nodes.new("ShaderNodeSeparateColor")
            rgb_node.name = rgb_name
            links.new(node.outputs["Color"], rgb_node.inputs["Color"])

        return rgb_node, output


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


def assign_mix_rgba(
    func: xc3_model_py.material.AssignmentFunc,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    blend_type: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend_type
    node.name = func_name(func)

    if blend_type == "OVERLAY":
        node.inputs["Factor"].default_value = 1.0

    assign_index(func.args[0], assignment_outputs, links, node.inputs["A"])
    assign_index(func.args[1], assignment_outputs, links, node.inputs["B"])
    if len(func.args) == 3:
        assign_index(func.args[2], assignment_outputs, links, node.inputs["Factor"])

    return node, "Result"


def assign_texture(
    texture: xc3_model_py.material.TextureAssignment,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures,
) -> Tuple[bpy.types.Node, str]:
    name = texture_assignment_name(texture)

    # Don't use the above name for node caching for any of the texture nodes.
    # This ensures the correct channel is assigned for each assignment.
    rgba_name = f"{name}.rgba"
    node = nodes.get(rgba_name)
    if node is None:
        node = import_texture(rgba_name, texture.name, nodes, textures)

    assign_uvs(texture.texcoords, assignment_outputs, node, nodes, links)

    return assign_channel(name, texture.channel, node, nodes, links)


def assign_uvs(texcoords: list[int], assignment_outputs, node, nodes, links):
    # Texture coordinates can be made of multiple nodes.
    uv_name = f"uv{texcoords}"
    uv_node = nodes.get(uv_name)
    if uv_node is None:
        uv_node = nodes.new("ShaderNodeCombineXYZ")
        uv_node.name = uv_name

        if len(texcoords) >= 2:
            assign_index(
                texcoords[0],
                assignment_outputs,
                links,
                uv_node.inputs["X"],
            )
            assign_index(
                texcoords[1],
                assignment_outputs,
                links,
                uv_node.inputs["Y"],
            )

    links.new(uv_node.outputs["Vector"], node.inputs["Vector"])


def import_texture(
    name: str,
    label: str,
    nodes,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
):
    node = nodes.new("ShaderNodeTexImage")
    node.name = name
    node.label = label

    if label in textures:
        image, sampler = textures[label]
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
    func: xc3_model_py.material.AssignmentFunc,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    op: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.new("ShaderNodeMath")
    node.operation = op
    node.name = func_name(func)

    for arg, input in zip(func.args, node.inputs):
        assign_index(arg, assignment_outputs, links, input)

    return node, "Value"


def func_name(func: xc3_model_py.material.AssignmentFunc):
    return func_name_inner(func.op, func.args)


def func_xyz_name(func: xc3_model_py.material.AssignmentFuncXyz):
    return func_name_inner(func.op, func.args)


def func_name_inner(op: xc3_model_py.shader_database.Operation, args: list[int]):
    op_name = str(op).removeprefix("Operation.")
    # Node groups that have multiple outputs can share a node.
    replacements = [
        ("AddNormalX", "AddNormal"),
        ("AddNormalY", "AddNormal"),
        ("ReflectX", "Reflect"),
        ("ReflectY", "Reflect"),
        ("ReflectZ", "Reflect"),
        ("NormalMapX", "NormalMap"),
        ("NormalMapY", "NormalMap"),
        ("NormalMapZ", "NormalMap"),
        ("MonochromeX", "Monochrome"),
        ("MonochromeY", "Monochrome"),
        ("MonochromeZ", "Monochrome"),
    ]
    for old, new in replacements:
        if op_name.startswith(old):
            op_name = op_name.replace(old, new)
            break

    func_args = ", ".join(str(a) for a in args)
    name = f"xyz_{op_name}({func_args})"
    return name


def texture_assignment_name(texture):
    coords = ", ".join(str(c) for c in texture.texcoords)
    return f"{texture.name}({coords})"


def assign_output_xyz(
    assignment_xyz: xc3_model_py.material.AssignmentXyz,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    assignment_outputs_xyz: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures: Dict[
        str, Tuple[Optional[bpy.types.Image], Optional[xc3_model_py.Sampler]]
    ],
) -> Optional[Tuple[bpy.types.Node, str]]:
    if func := assignment_xyz.func():
        return assign_func_xyz(func, assignment_outputs_xyz, nodes, links)
    elif value := assignment_xyz.value():
        return assign_value_xyz(value, assignment_outputs, nodes, links, textures)


def assign_func_xyz(
    func: xc3_model_py.material.AssignmentFuncXyz,
    assignment_outputs_xyz: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str]]:
    mix_rgba_node = lambda ty: assign_mix_xyz(
        func,
        assignment_outputs_xyz,
        nodes,
        links,
        ty,
    )

    math_node = lambda ty: assign_math_xyz(
        func,
        assignment_outputs_xyz,
        nodes,
        links,
        ty,
    )

    group_node = lambda func, name, create_node_tree: create_cached_func_xyz_group_node(
        nodes, func, name, create_node_tree
    )

    assign_args = lambda func, node, params: assign_func_args(
        func, params, assignment_outputs_xyz, links, node
    )

    match func.op:
        case xc3_model_py.shader_database.Operation.Unk:
            return None
        case xc3_model_py.shader_database.Operation.Mix:
            return mix_rgba_node("MIX")
        case xc3_model_py.shader_database.Operation.Mul:
            return math_node("MULTIPLY")
        case xc3_model_py.shader_database.Operation.Div:
            return math_node("DIVIDE")
        case xc3_model_py.shader_database.Operation.Add:
            return math_node("ADD")
        case xc3_model_py.shader_database.Operation.Sub:
            return math_node("SUBTRACT")
        case xc3_model_py.shader_database.Operation.Fma:
            return math_node("MULTIPLY_ADD")
        case xc3_model_py.shader_database.Operation.MulRatio:
            node = nodes.new("ShaderNodeMix")
            node.data_type = "RGBA"
            node.blend_type = "MULTIPLY"
            node.name = func_xyz_name(func)
            assign_args(func, node, ["A", "B", "Factor"])
            return node, "Result"
        case xc3_model_py.shader_database.Operation.AddNormalX:
            pass
        case xc3_model_py.shader_database.Operation.AddNormalY:
            pass
        case xc3_model_py.shader_database.Operation.Overlay:
            return mix_rgba_node("OVERLAY")
        case xc3_model_py.shader_database.Operation.Overlay2:
            return mix_rgba_node("OVERLAY")
        case xc3_model_py.shader_database.Operation.OverlayRatio:
            return mix_rgba_node("OVERLAY")
        case xc3_model_py.shader_database.Operation.Power:
            node = group_node(func, "PowerXYZ", power_xyz_node_group)
            assign_args(func, node, ["Base", "Exponent"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.Min:
            return math_node("MINIMUM")
        case xc3_model_py.shader_database.Operation.Max:
            return math_node("MAXIMUM")
        case xc3_model_py.shader_database.Operation.Clamp:
            node = group_node(func, "ClampXYZ", clamp_xyz_node_group)
            assign_args(func, node, ["Value", "Min", "Max"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.Abs:
            return math_node("ABSOLUTE")
        case xc3_model_py.shader_database.Operation.Fresnel:
            # TODO: Separate factors for xyz?
            node = group_node(func, "FresnelBlend", fresnel_blend_node_group)
            # TODO: normals?
            assign_args(func, node, ["Value"])
            return node, "Value"
        case xc3_model_py.shader_database.Operation.Sqrt:
            node = group_node(func, "SqrtXYZ", sqrt_xyz_node_group)
            assign_args(func, node, ["Value"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.TexMatrix:
            pass
        case xc3_model_py.shader_database.Operation.TexParallaxX:
            pass
        case xc3_model_py.shader_database.Operation.TexParallaxY:
            pass
        case xc3_model_py.shader_database.Operation.ReflectX:
            pass
        case xc3_model_py.shader_database.Operation.ReflectY:
            pass
        case xc3_model_py.shader_database.Operation.ReflectZ:
            pass
        case xc3_model_py.shader_database.Operation.Floor:
            return math_node("FLOOR")
        case xc3_model_py.shader_database.Operation.Select:
            return mix_rgba_node("MIX")
        case xc3_model_py.shader_database.Operation.Equal:
            return math_node("COMPARE")
        case xc3_model_py.shader_database.Operation.NotEqual:
            # TODO: Invert compare.
            pass
        case xc3_model_py.shader_database.Operation.Less:
            node = group_node(nodes, "LessXYZ", less_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.Greater:
            node = group_node(nodes, "GreaterXYZ", greater_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.LessEqual:
            # TODO: node group for leq?
            node = group_node(nodes, "LessXYZ", less_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.GreaterEqual:
            # TODO: node group for geq?
            node = group_node(nodes, "GreaterXYZ", greater_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.Dot4:
            pass
        case xc3_model_py.shader_database.Operation.NormalMapX:
            pass
        case xc3_model_py.shader_database.Operation.NormalMapY:
            pass
        case xc3_model_py.shader_database.Operation.NormalMapZ:
            pass
        # TODO: Fix merging for monochrome xyz?
        case xc3_model_py.shader_database.Operation.Negate:
            node = nodes.new("ShaderNodeVectorMath")
            node.name = func_xyz_name(func)
            node.operation = "MULTIPLY"
            assign_args(func, node, [0])
            node.inputs[1].default_value = (-1.0, -1.0, -1.0)
            return node, "Vector"
        case xc3_model_py.shader_database.Operation.FurInstanceAlpha:
            node = nodes.new("ShaderNodeAttribute")
            node.name = func_name(func)
            node.attribute_name = "FurAlpha"
            return node, "Fac"
        case _:
            # TODO: This case shouldn't happen?
            return None


def assign_mix_xyz(
    func: xc3_model_py.material.AssignmentFuncXyz,
    assignment_outputs_xyz: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    blend_type: str,
) -> Tuple[bpy.types.Node, str]:
    # TODO: Custom nodes with vector math to support negative values?
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend_type
    node.name = func_xyz_name(func)

    if blend_type == "OVERLAY":
        node.inputs["Factor"].default_value = 1.0

    assign_index(
        func.args[0],
        assignment_outputs_xyz,
        links,
        node.inputs["A"],
    )
    assign_index(
        func.args[1],
        assignment_outputs_xyz,
        links,
        node.inputs["B"],
    )
    if len(func.args) == 3:
        assign_index(
            func.args[2],
            assignment_outputs_xyz,
            links,
            node.inputs["Factor"],
        )

    return node, "Result"


def assign_math_xyz(
    func,
    assignments_xyz: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    op: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = op
    node.name = func_xyz_name(func)

    for arg, input in zip(func.args, node.inputs):
        assign_index(
            arg,
            assignments_xyz,
            links,
            input,
        )

    return node, "Vector"


def assign_value_xyz(
    value: xc3_model_py.material.AssignmentValueXyz,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures,
) -> Optional[Tuple[bpy.types.Node, str]]:
    if floats := value.float():
        if all(f >= 0.0 and f <= 1.0 for f in floats):
            # Use an RGB node if possible to show a preview color.
            node = nodes.new("ShaderNodeRGB")
            node.outputs[0].default_value = (floats[0], floats[1], floats[2], 1.0)
            return node, "Color"
        else:
            node = nodes.new("ShaderNodeCombineXYZ")
            node.inputs["X"].default_value = floats[0]
            node.inputs["Y"].default_value = floats[1]
            node.inputs["Z"].default_value = floats[2]
            return node, "Vector"
    elif attribute := value.attribute():
        return assign_attribute_xyz(attribute, nodes, links)
    elif texture := value.texture():
        return assign_texture_xyz(texture, assignment_outputs, nodes, links, textures)
    else:
        return None


def assign_attribute_xyz(
    attribute: xc3_model_py.material.AssignmentValueAttributeXyz, nodes, links
) -> Optional[Tuple[bpy.types.Node, str]]:
    node = import_attribute(attribute.name, nodes)
    return assign_channel_xyz(attribute.name, attribute.channel, node, nodes, links)


def import_attribute(name: str, nodes) -> bpy.types.Node:
    node = nodes.get(name)
    if node is None:
        node = nodes.new("ShaderNodeAttribute")
        node.name = name

        if name == "vPos":
            node.attribute_name = "position"
        elif name == "vNormal":
            node.attribute_name = "VertexNormal"
        elif name == "vColor":
            node.attribute_name = "VertexColor"
        elif name == "vBlend":
            node.attribute_name = "Blend"
        else:
            for i in range(9):
                if name == f"vTex{i}":
                    node.attribute_name = f"TexCoord{i}"
                    break
    return node


def assign_channel_xyz(
    name: str,
    channel: Optional[xc3_model_py.material.ChannelXyz],
    node,
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str]]:
    match channel:
        case xc3_model_py.material.ChannelXyz.Xyz:
            return node, "Color"
        case xc3_model_py.material.ChannelXyz.X:
            return assign_channel(name, "Red", node, nodes, links)
        case xc3_model_py.material.ChannelXyz.Y:
            return assign_channel(name, "Green", node, nodes, links)
        case xc3_model_py.material.ChannelXyz.Z:
            return assign_channel(name, "Blue", node, nodes, links)
        case xc3_model_py.material.ChannelXyz.W:
            return assign_channel(name, "Alpha", node, nodes, links)
        case _:
            return node, "Color"


# TODO: Share code with scalar version.
def assign_texture_xyz(
    texture: xc3_model_py.material.TextureAssignmentXyz,
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures,
) -> Optional[Tuple[bpy.types.Node, str]]:
    name = texture_assignment_name(texture)

    # Don't use the above name for node caching for any of the texture nodes.
    # This ensures the correct channel is assigned for each assignment.
    rgba_name = f"{name}.rgba"
    node = nodes.get(rgba_name)
    if node is None:
        node = import_texture(rgba_name, texture.name, nodes, textures)

    # XYZ texture coordinate assignments still use the regular scalar assignments.
    assign_uvs(texture.texcoords, assignment_outputs, node, nodes, links)

    return assign_channel_xyz(name, texture.channel, node, nodes, links)


def used_assignments(
    output_assignments: xc3_model_py.material.OutputAssignments, has_alpha: bool
) -> Set[int]:
    visited = set()

    assignments = output_assignments.assignments
    outputs = output_assignments.output_assignments

    if xyz := outputs[0].merge_xyz(assignments):
        add_used_xyz_assignments(visited, assignments, xyz.assignments, xyz.assignment)
    else:
        add_used_assignments(visited, assignments, outputs[0].x)
        add_used_assignments(visited, assignments, outputs[0].y)
        add_used_assignments(visited, assignments, outputs[0].z)

    if has_alpha:
        add_used_assignments(visited, assignments, outputs[0].w)

    add_used_assignments(visited, assignments, outputs[1].x)
    add_used_assignments(visited, assignments, outputs[1].y)

    add_used_assignments(visited, assignments, outputs[2].x)
    add_used_assignments(visited, assignments, outputs[2].y)
    add_used_assignments(visited, assignments, outputs[2].z)

    if xyz := outputs[5].merge_xyz(assignments):
        add_used_xyz_assignments(visited, assignments, xyz.assignments, xyz.assignment)
    else:
        add_used_assignments(visited, assignments, outputs[5].x)
        add_used_assignments(visited, assignments, outputs[5].y)
        add_used_assignments(visited, assignments, outputs[5].z)

    return visited


def add_used_assignments(
    visited: Set[int],
    assignments: list[xc3_model_py.material.Assignment],
    i: Optional[int],
):
    if i is not None:
        if i not in visited:
            visited.add(i)

            assignment = assignments[i]
            if func := assignment.func():
                for arg in func.args:
                    add_used_assignments(visited, assignments, arg)
            elif value := assignment.value():
                if texture := value.texture():
                    for coord in texture.texcoords:
                        add_used_assignments(visited, assignments, coord)


def add_used_xyz_assignments(
    visited: Set[int],
    assignments: list[xc3_model_py.material.Assignment],
    assignments_xyz: list[xc3_model_py.material.AssignmentXyz],
    i: int,
):
    # Collect the scalar assignments for texture coordinates.
    if i is not None:
        assignment = assignments_xyz[i]
        if func := assignment.func():
            for arg in func.args:
                add_used_xyz_assignments(visited, assignments, assignments_xyz, arg)
        elif value := assignment.value():
            if texture := value.texture():
                for coord in texture.texcoords:
                    add_used_assignments(visited, assignments, coord)


def create_cached_func_group_node(
    nodes,
    func: xc3_model_py.material.AssignmentFunc,
    node_group_name: str,
    create_node_tree,
) -> bpy.types.Node:
    name = func_name(func)
    node = nodes.get(name)
    if node is None:
        node = create_node_group(nodes, node_group_name, create_node_tree)
        node.name = name

    return node


def assign_func_args(
    func: xc3_model_py.material.AssignmentFunc,
    params: list[int | str],
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    links,
    node: bpy.types.Node,
):
    for i, param in zip(func.args, params):
        assign_index(i, assignment_outputs, links, node.inputs[param])


def create_cached_func_xyz_group_node(
    nodes,
    func: xc3_model_py.material.AssignmentFuncXyz,
    node_group_name: str,
    create_node_tree,
) -> bpy.types.Node:
    name = func_xyz_name(func)
    node = nodes.get(name)
    if node is None:
        node = create_node_group(nodes, node_group_name, create_node_tree)
        node.name = name

    return node
