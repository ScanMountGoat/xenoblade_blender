import bpy

from xenoblade_blender.node_layout import layout_nodes


def create_node_group(nodes, name: str, create_node_tree):
    # Cache the node group creation.
    node_tree = bpy.data.node_groups.get(name)
    if node_tree is None:
        node_tree = create_node_tree()

    group = nodes.new("ShaderNodeGroup")
    group.node_tree = node_tree
    return group


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

    layout_nodes(output_node, links)

    return node_tree


def add_normals_node_group():
    node_tree = bpy.data.node_groups.new("AddNormals", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Factor"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.y"
    )

    normal_a = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
    normal_a.inputs["X"].default_value = 0.5
    normal_a.inputs["Y"].default_value = 0.5
    links.new(input_node.outputs["A.x"], normal_a.inputs["X"])
    links.new(input_node.outputs["A.y"], normal_a.inputs["Y"])

    normal_b = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
    normal_b.inputs["X"].default_value = 0.5
    normal_b.inputs["Y"].default_value = 0.5
    links.new(input_node.outputs["B.x"], normal_b.inputs["X"])
    links.new(input_node.outputs["B.y"], normal_b.inputs["Y"])

    t = nodes.new("ShaderNodeVectorMath")
    t.operation = "ADD"
    t.inputs[1].default_value = (0.0, 0.0, 1.0)
    links.new(normal_a.outputs["Normal"], t.inputs[0])

    u = nodes.new("ShaderNodeVectorMath")
    u.operation = "MULTIPLY"
    u.inputs[1].default_value = (-1.0, -1.0, 1.0)
    links.new(normal_b.outputs["Normal"], u.inputs[0])

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
    links.new(normal_a.outputs["Normal"], mix.inputs["A"])
    links.new(normalize_r.outputs["Vector"], mix.inputs["B"])

    normalize_result = nodes.new("ShaderNodeVectorMath")
    normalize_result.operation = "NORMALIZE"
    links.new(mix.outputs["Result"], normalize_result.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")

    # Remap to the 0.0 to 1.0 range to make blending easier.
    remap_normals = nodes.new("ShaderNodeVectorMath")
    remap_normals.operation = "MULTIPLY_ADD"
    links.new(normalize_result.outputs["Vector"], remap_normals.inputs["Vector"])
    remap_normals.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap_normals.inputs[2].default_value = (0.5, 0.5, 0.5)

    output_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(remap_normals.outputs["Vector"], output_xyz.inputs["Vector"])
    links.new(output_xyz.outputs["X"], output_node.inputs["X"])
    links.new(output_xyz.outputs["Y"], output_node.inputs["Y"])

    layout_nodes(output_node, links)

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

    layout_nodes(output_node, links)

    return node_tree


def tex_matrix_node_group():
    node_tree = bpy.data.node_groups.new("TexMatrix", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="U"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="V"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="C"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="D"
    )

    # dot(vec4(u, v, 0, 1), vec4(a, b, c, d)) == dot(vec3(u, v, 1), vec3(a, b, d))
    uv = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["U"], uv.inputs["X"])
    links.new(input_node.outputs["V"], uv.inputs["Y"])
    uv.inputs["Z"].default_value = 1.0

    abd = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["A"], abd.inputs["X"])
    links.new(input_node.outputs["B"], abd.inputs["Y"])
    links.new(input_node.outputs["D"], abd.inputs["Z"])

    dot_uv = nodes.new("ShaderNodeVectorMath")
    dot_uv.operation = "DOT_PRODUCT"
    links.new(uv.outputs["Vector"], dot_uv.inputs[0])
    links.new(abd.outputs["Vector"], dot_uv.inputs[1])

    output_node = nodes.new("NodeGroupOutput")
    links.new(dot_uv.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def tex_parallax_node_group():
    node_tree = bpy.data.node_groups.new("TexParallax", "ShaderNodeTree")

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
        in_out="INPUT", socket_type="NodeSocketFloat", name="Factor"
    )

    # TODO: Implement this properly

    output_node = nodes.new("NodeGroupOutput")
    links.new(input_node.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def normal_map_xyz_node_group():
    node_tree = bpy.data.node_groups.new("NormalMapXY", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
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
    # TODO: Strength?

    normals = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
    links.new(input_node.outputs["X"], normals.inputs["X"])
    links.new(input_node.outputs["Y"], normals.inputs["Y"])

    remap_normals = nodes.new("ShaderNodeVectorMath")
    remap_normals.operation = "MULTIPLY_ADD"
    links.new(normals.outputs["Normal"], remap_normals.inputs[0])
    remap_normals.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap_normals.inputs[2].default_value = (0.5, 0.5, 0.5)

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(remap_normals.outputs["Vector"], normal_map.inputs["Color"])

    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(normal_map.outputs["Normal"], xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(xyz.outputs["X"], output_node.inputs["X"])
    links.new(xyz.outputs["Y"], output_node.inputs["Y"])
    links.new(xyz.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def normal_map_xy_final_node_group():
    node_tree = bpy.data.node_groups.new("NormalMapXYFinal", "ShaderNodeTree")

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
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Strength"
    )

    normals = create_node_group(nodes, "NormalsXY", normals_xy_node_group)
    links.new(input_node.outputs["X"], normals.inputs["X"])
    links.new(input_node.outputs["Y"], normals.inputs["Y"])

    remap_normals = nodes.new("ShaderNodeVectorMath")
    remap_normals.operation = "MULTIPLY_ADD"
    links.new(normals.outputs["Normal"], remap_normals.inputs[0])
    remap_normals.inputs[1].default_value = (0.5, 0.5, 0.5)
    remap_normals.inputs[2].default_value = (0.5, 0.5, 0.5)

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(remap_normals.outputs["Vector"], normal_map.inputs["Color"])
    links.new(input_node.outputs["Strength"], normal_map.inputs["Strength"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(normal_map.outputs["Normal"], output_node.inputs["Normal"])

    layout_nodes(output_node, links)

    return node_tree


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

    layout_nodes(output_node, links)

    return node_tree


def reflect_xyz_node_group():
    node_tree = bpy.data.node_groups.new("ReflectXYZ", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Z"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.z"
    )

    a = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["A.x"], a.inputs["X"])
    links.new(input_node.outputs["A.y"], a.inputs["Y"])
    links.new(input_node.outputs["A.z"], a.inputs["Z"])

    b = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["B.x"], b.inputs["X"])
    links.new(input_node.outputs["B.y"], b.inputs["Y"])
    links.new(input_node.outputs["B.z"], b.inputs["Z"])

    reflect = nodes.new("ShaderNodeVectorMath")
    reflect.operation = "REFLECT"
    links.new(a.outputs["Vector"], reflect.inputs[0])
    links.new(b.outputs["Vector"], reflect.inputs[1])

    output_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(reflect.outputs["Vector"], output_xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(output_xyz.outputs["X"], output_node.inputs["X"])
    links.new(output_xyz.outputs["Y"], output_node.inputs["Y"])
    links.new(output_xyz.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def power_xyz_node_group():
    return math_xyz_node_group("PowerXYZ", "POWER", ["Base", "Exponent"])


def sqrt_xyz_node_group():
    return math_xyz_node_group("SqrtXYZ", "SQRT", ["Value"])


def less_xyz_node_group():
    return math_xyz_node_group("LessXYZ", "LESS_THAN", ["Value", "Threshold"])


def greater_xyz_node_group():
    return math_xyz_node_group("GreaterXYZ", "GREATER_THAN", ["Value", "Threshold"])


def clamp_xyz_node_group():
    node_tree = bpy.data.node_groups.new("ClampXYZ", "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Value"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Min"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Max"
    )

    min_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Min"], min_xyz.inputs["Vector"])

    max_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Max"], max_xyz.inputs["Vector"])

    value_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Value"], value_xyz.inputs["Vector"])

    clamp_x = nodes.new("ShaderNodeClamp")
    links.new(value_xyz.outputs["X"], clamp_x.inputs["Value"])
    links.new(min_xyz.outputs["X"], clamp_x.inputs["Min"])
    links.new(max_xyz.outputs["X"], clamp_x.inputs["Max"])

    clamp_y = nodes.new("ShaderNodeClamp")
    links.new(value_xyz.outputs["Y"], clamp_y.inputs["Value"])
    links.new(min_xyz.outputs["Y"], clamp_y.inputs["Min"])
    links.new(max_xyz.outputs["Y"], clamp_y.inputs["Max"])

    clamp_z = nodes.new("ShaderNodeClamp")
    links.new(value_xyz.outputs["Z"], clamp_z.inputs["Value"])
    links.new(min_xyz.outputs["Z"], clamp_z.inputs["Min"])
    links.new(max_xyz.outputs["Z"], clamp_z.inputs["Max"])

    output_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(clamp_x.outputs["Result"], output_xyz.inputs["X"])
    links.new(clamp_y.outputs["Result"], output_xyz.inputs["Y"])
    links.new(clamp_z.outputs["Result"], output_xyz.inputs["Z"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(output_xyz.outputs["Vector"], output_node.inputs["Vector"])

    layout_nodes(output_node, links)

    return node_tree


def math_xyz_node_group(name: str, op: str, inputs: list[str]):
    # Apply a scalar operation to independent XYZ components.
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    for i in inputs:
        node_tree.interface.new_socket(
            in_out="INPUT", socket_type="NodeSocketVector", name=i
        )

    input_xyz_nodes = []
    for i in inputs:
        node = nodes.new("ShaderNodeSeparateXYZ")
        links.new(input_node.outputs[i], node.inputs["Vector"])
        input_xyz_nodes.append(node)

    op_x = nodes.new("ShaderNodeMath")
    op_x.operation = op
    for i, node in enumerate(input_xyz_nodes):
        links.new(node.outputs["X"], op_x.inputs[i])

    op_y = nodes.new("ShaderNodeMath")
    op_y.operation = op
    for i, node in enumerate(input_xyz_nodes):
        links.new(node.outputs["Y"], op_y.inputs[i])

    op_z = nodes.new("ShaderNodeMath")
    op_z.operation = op
    for i, node in enumerate(input_xyz_nodes):
        links.new(node.outputs["Z"], op_z.inputs[i])

    output_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(op_x.outputs["Value"], output_xyz.inputs["X"])
    links.new(op_y.outputs["Value"], output_xyz.inputs["Y"])
    links.new(op_z.outputs["Value"], output_xyz.inputs["Z"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(output_xyz.outputs["Vector"], output_node.inputs["Vector"])

    layout_nodes(output_node, links)

    return node_tree
