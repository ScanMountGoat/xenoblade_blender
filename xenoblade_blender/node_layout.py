from typing import Tuple
import bpy


def layout_nodes(root: bpy.types.Node):
    # Assign each node to a layer.
    layers = []
    node_layer = {}
    assign_node_layers(root, layers, 0, node_layer)

    # Basic layered graph layout.
    margin_x = 100
    margin_y = 20

    location_x = 0
    for i, nodes in enumerate(layers):
        if i > 0:
            layer_max_width = max(node_dimensions(n)[0] for n in nodes)
            location_x -= layer_max_width + margin_x

        location_y = 0
        for n in nodes:
            n.location.x = location_x
            n.location.y = location_y

            location_y -= node_dimensions(n)[1] + margin_y


def assign_node_layers(
    node: bpy.types.Node,
    layers: list[list[bpy.types.Node]],
    layer: int,
    node_layer: dict[str, int],
):
    # Assign each node to only one layer.
    # Assign to the deepest layer to make edges go from left to right.
    # This assumes the graph is acyclic.
    if previous_layer := node_layer.get(node.name):
        if layer <= previous_layer:
            return

        layers[previous_layer].remove(node)

    node_layer[node.name] = layer

    if layer >= len(layers):
        layers.append([])

    layers[layer].append(node)

    for input in node.inputs:
        if input.is_linked:
            for link in input.links:
                assign_node_layers(link.from_node, layers, layer + 1, node_layer)


def node_dimensions(node: bpy.types.Node) -> Tuple[float, float]:
    # Width and height aren't updated until nodes are drawn on screen.
    # Guess the final dimensions with a default of (140, 100).
    default_dimensions = {
        "ShaderNodeBsdfPrincipled": (240.0, 340.0),
        "ShaderNodeOutputMaterial": (140.0, 140.0),
        "ShaderNodeTexImage": (240.0, 271.0),
        "ShaderNodeCombineColor": (140.0, 143.0),
        "ShaderNodeMix": (140.0, 222.0),
        "ShaderNodeSeparateColor": (140.0, 143.0),
        "ShaderNodeUVMap": (150.0, 104.0),
        "ShaderNodeVectorMath": (140.0, 271.0),
        "ShaderNodeMath": (140.0, 148.0),
        "ShaderNodeGroup": (140.0, 118.0),
        "ShaderNodeNormalMap": (150.0, 146.0),
        "ShaderNodeValue": (140.0, 79.0),
        "ShaderNodeAttribute": (140.0, 170.0),
    }

    return default_dimensions.get(node.bl_idname, (140.0, 100.0))
