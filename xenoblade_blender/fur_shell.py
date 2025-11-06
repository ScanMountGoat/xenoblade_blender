import bpy
import typing

from xenoblade_blender.node_layout import layout_nodes

if typing.TYPE_CHECKING:
    from ..xc3_model_py.xc3_model_py import xc3_model_py
else:
    from . import xc3_model_py


def import_fur_shells(
    mesh: bpy.types.Object, fur_params: xc3_model_py.material.FurShellParams
):
    name = "FurShellGeometryNodes"
    modifier = mesh.modifiers.new(name, type="NODES")

    node_tree = bpy.data.node_groups.get(name)
    if node_tree is None:
        # Recreate the vertex shader operations and parameters for the uniform buffer.
        node_tree = fur_shell_geometry_node_group(name)

    modifier.node_group = node_tree

    set_modifier_property(modifier, "Count", fur_params.instance_count)
    set_modifier_property(modifier, "Scale", fur_params.shell_width)
    set_modifier_property(modifier, "Height Offset", fur_params.y_offset)
    set_modifier_property(modifier, "Alpha", fur_params.alpha)


def set_modifier_property(modifier: bpy.types.Modifier, name: str, value):
    # Inputs don't use the specified name.
    id = modifier.node_group.interface.items_tree[name].identifier
    modifier[id] = value


def fur_shell_geometry_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "GeometryNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketGeometry", name="Geometry"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="FurAlpha"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketGeometry", name="Geometry"
    )
    # TODO: Better names?
    count_socket = node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketInt", name="Count"
    )
    count_socket.min_value = 0
    count_socket.max_value = 20  # TODO: Find a good max value in game.

    scale_socket = node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Scale"
    )
    scale_socket.min_value = 0.0
    scale_socket.max_value = 1.0  # TODO: Find a good max value in game.

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Height Offset"
    )
    alpha_socket = node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Alpha"
    )
    alpha_socket.min_value = 0.0
    alpha_socket.max_value = 1.0

    # Create N instances without using nodes from newer Blender versions.
    grid = nodes.new("GeometryNodeMeshGrid")
    grid.inputs[0].default_value = 0.0
    grid.inputs[1].default_value = 0.0
    links.new(input_node.outputs["Count"], grid.inputs[2])
    grid.inputs[3].default_value = 1

    instances = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(grid.outputs["Mesh"], instances.inputs["Points"])
    links.new(input_node.outputs["Geometry"], instances.inputs["Instance"])

    capture_index = nodes.new("GeometryNodeCaptureAttribute")
    capture_index.domain = "INSTANCE"
    try:
        # Multiple captures was added in Blender 4.2.
        capture_index.capture_items.new("INT", "Index")
    except:
        # This is only present in 4.1.
        capture_index.data_type = "INT"

    links.new(instances.outputs["Instances"], capture_index.inputs[0])

    index = nodes.new("GeometryNodeInputIndex")
    links.new(index.outputs["Index"], capture_index.inputs[1])

    realize_instances = nodes.new("GeometryNodeRealizeInstances")
    links.new(capture_index.outputs["Geometry"], realize_instances.inputs["Geometry"])

    # Uniform buffer shell width parameter.
    one_over_count = nodes.new("ShaderNodeMath")
    one_over_count.operation = "DIVIDE"
    one_over_count.inputs[0].default_value = 1.0
    links.new(input_node.outputs["Count"], one_over_count.inputs[1])

    shell_width_param = nodes.new("ShaderNodeMath")
    shell_width_param.operation = "MULTIPLY"
    links.new(input_node.outputs["Scale"], shell_width_param.inputs[0])
    links.new(one_over_count.outputs["Value"], shell_width_param.inputs[1])

    # Vertex shader instance "scale" by translating along normal.
    index_plus_one = nodes.new("ShaderNodeMath")
    index_plus_one.operation = "ADD"
    links.new(capture_index.outputs[1], index_plus_one.inputs[0])
    index_plus_one.inputs[1].default_value = 1.0

    instance_scale = nodes.new("ShaderNodeMath")
    instance_scale.operation = "MULTIPLY"
    links.new(index_plus_one.outputs["Value"], instance_scale.inputs[0])
    links.new(shell_width_param.outputs["Value"], instance_scale.inputs[1])

    scale_normal = nodes.new("ShaderNodeVectorMath")
    scale_normal.operation = "SCALE"
    links.new(instance_scale.outputs["Value"], scale_normal.inputs["Scale"])

    normal = nodes.new("GeometryNodeInputNormal")
    links.new(normal.outputs["Normal"], scale_normal.inputs[0])

    # Uniform buffer height offset parameter.
    height_offset_param = nodes.new("ShaderNodeMath")
    height_offset_param.operation = "MULTIPLY"
    links.new(input_node.outputs["Scale"], height_offset_param.inputs[0])
    links.new(input_node.outputs["Height Offset"], height_offset_param.inputs[1])

    # Vertex shader instance height offset.
    y_offset_param = nodes.new("ShaderNodeMath")
    y_offset_param.operation = "DIVIDE"
    links.new(index_plus_one.outputs["Value"], y_offset_param.inputs[0])
    links.new(input_node.outputs["Count"], y_offset_param.inputs[1])

    y_offset_param3 = nodes.new("ShaderNodeMath")
    y_offset_param3.operation = "POWER"
    links.new(y_offset_param.outputs["Value"], y_offset_param3.inputs[0])
    y_offset_param3.inputs[1].default_value = 3.0

    y_offset = nodes.new("ShaderNodeMath")
    y_offset.operation = "MULTIPLY"
    links.new(y_offset_param3.outputs["Value"], y_offset.inputs[0])
    links.new(height_offset_param.outputs["Value"], y_offset.inputs[1])

    add_y_offset = nodes.new("ShaderNodeVectorMath")
    add_y_offset.operation = "ADD"
    links.new(scale_normal.outputs["Vector"], add_y_offset.inputs[0])

    # Convert y-up in game to z-up in Blender.
    xyz_offset = nodes.new("ShaderNodeCombineXYZ")
    links.new(y_offset.outputs["Value"], xyz_offset.inputs["Z"])
    links.new(xyz_offset.outputs["Vector"], add_y_offset.inputs[1])

    apply_scale = nodes.new("GeometryNodeSetPosition")
    links.new(realize_instances.outputs["Geometry"], apply_scale.inputs["Geometry"])
    links.new(add_y_offset.outputs["Vector"], apply_scale.inputs["Offset"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(apply_scale.outputs["Geometry"], output_node.inputs["Geometry"])

    # Uniform buffer alpha parameter.
    one_minus_alpha = nodes.new("ShaderNodeMath")
    one_minus_alpha.operation = "SUBTRACT"
    one_minus_alpha.inputs[0].default_value = 1.0
    links.new(input_node.outputs["Alpha"], one_minus_alpha.inputs[1])

    alpha_param = nodes.new("ShaderNodeMath")
    alpha_param.operation = "DIVIDE"
    links.new(one_minus_alpha.outputs["Value"], alpha_param.inputs[0])
    links.new(input_node.outputs["Count"], alpha_param.inputs[1])

    # Vertex shader instance alpha.
    alpha_factor = nodes.new("ShaderNodeMath")
    alpha_factor.operation = "MULTIPLY"
    links.new(capture_index.outputs[1], alpha_factor.inputs[0])
    links.new(alpha_param.outputs["Value"], alpha_factor.inputs[1])
    alpha_factor.use_clamp = True

    one_minus_alpha_factor = nodes.new("ShaderNodeMath")
    one_minus_alpha_factor.operation = "SUBTRACT"
    one_minus_alpha_factor.inputs[0].default_value = 1.0
    links.new(alpha_factor.outputs["Value"], one_minus_alpha_factor.inputs[1])

    # TODO: Modify vertex alpha instead.
    links.new(one_minus_alpha_factor.outputs["Value"], output_node.inputs["FurAlpha"])

    layout_nodes(output_node, links)

    return node_tree
