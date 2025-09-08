import logging
from typing import Dict, Optional
import bpy
import numpy as np
import os
import math

from .import_material import import_material
from . import xc3_model_py

from mathutils import Matrix
from bpy_extras import image_utils
from pathlib import Path


def get_database_path() -> str:
    return os.path.join(os.path.dirname(__file__), "xc_combined.bin")


def get_image_folder(image_folder: str, filepath: str) -> str:
    if image_folder == "":
        return str(Path(filepath).parent)
    else:
        return image_folder


def init_logging():
    # Log any errors from Rust.
    log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.ERROR)


# https://github.com/ssbucarlos/smash-ultimate-blender/blob/a003be92bd27e34d2a6377bb98d55d5a34e63e56/source/model/import_model.py#L371
def import_armature(operator, context, root, name: str):
    armature = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.collection.objects.link(armature)

    armature.data.display_type = "STICK"
    armature.rotation_mode = "QUATERNION"
    armature.show_in_front = True

    previous_active = context.view_layer.objects.active
    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT", toggle=False)

    if root.skeleton is not None:
        transforms = root.skeleton.model_space_transforms()

        for bone, transform in zip(root.skeleton.bones, transforms):
            new_bone = armature.data.edit_bones.new(name=bone.name)
            new_bone.head = [0, 0, 0]
            new_bone.tail = [0, 1, 0]
            matrix = Matrix(transform)
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
            # Prevent Blender from removing any bones.
            bone.length = 0.1

    elif root.buffers.weights is not None:
        message = "Unable to load skeleton for model with skin weights."
        message += " The .arc or .chr file is missing or does not contain bone data."
        operator.report({"WARNING"}, message)

    bpy.ops.object.mode_set(mode="OBJECT")
    context.view_layer.objects.active = previous_active

    return armature


def merge_armatures(operator, context, armatures):
    combined_armature = None
    if context.object is not None and isinstance(
        context.object.data, bpy.types.Armature
    ):
        # Merge to the selected armature if present.
        combined_armature = context.object
    elif len(armatures) > 1:
        # Merge the supplied armatures if none are selected.
        combined_armature = armatures[0]

    if combined_armature is None:
        message = (
            "Skipping armature merging. No armature selected or importing single file."
        )
        operator.report({"WARNING"}, message)
        return

    previous_active = context.view_layer.objects.active
    context.view_layer.objects.active = combined_armature
    bpy.ops.object.mode_set(mode="EDIT", toggle=False)

    combined_bones = combined_armature.data.edit_bones

    # Merge each bone instead of finding the armature with more bones.
    # This is necessary for some split models to animate correctly.
    for armature in armatures:
        for bone in armature.data.bones:
            if bone.name not in combined_bones:
                # Create a copy of the bone.
                # This works since bone.matrix is relative to the parent.
                new_bone = combined_bones.new(name=bone.name)
                new_bone.head = [0, 0, 0]
                new_bone.tail = [0, 1, 0]
                new_bone.matrix = bone.matrix_local

        # Update parenting once all new bones are added.
        for bone in armature.data.bones:
            if bone.parent is not None:
                combined_bone = combined_bones.get(bone.name)
                combined_bone.parent = combined_bones.get(bone.parent.name)

                # Prevent Blender from removing any bones.
                combined_bone.length = 0.1

    bpy.ops.object.mode_set(mode="OBJECT")
    context.view_layer.objects.active = previous_active

    # Apply the armature to all models.
    for armature in armatures:
        for o in armature.children:
            o.parent = combined_armature
            for modifier in o.modifiers:
                if modifier.type == "ARMATURE":
                    modifier.object = combined_armature


def import_images(
    root, model_name: str, pack: bool, image_folder: str, flip: bool
) -> list[bpy.types.Image]:
    blender_images = []

    if pack:
        png_images = xc3_model_py.decode_images_png(root.image_textures, not flip)
        for i, (image, png) in enumerate(zip(root.image_textures, png_images)):
            blender_image = import_image(image, png, model_name, i)
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


def import_image(image, png: bytes, model_name: str, i: int):
    # Use the same naming conventions as the saved PNG images and xc3_tex.
    if model_name != "":
        if image.name is not None:
            name = f"{model_name}.{i}.{image.name}"
        else:
            name = f"{model_name}.{i}"
    else:
        name = image.name

    blender_image = bpy.data.images.new(name, image.width, image.height, alpha=True)

    # Loading png bytes is faster than foreach_set with a float buffer.
    blender_image.pack(data=png, data_len=len(png))
    blender_image.source = "FILE"

    # TODO: This should depend on srgb vs linear in format.
    blender_image.colorspace_settings.is_data = True

    # Necessary for 0 alpha to not set RGB to black.
    blender_image.alpha_mode = "CHANNEL_PACKED"

    return blender_image


def import_monolib_shader_images(
    path: str, flip: bool
) -> Optional[Dict[str, bpy.types.Image]]:
    # Assume the path is in a game dump.
    for parent in Path(path).parents:
        folder = parent.joinpath("monolib").joinpath("shader")
        if folder.exists():
            # TODO: Lazy load these images instead?
            textures = xc3_model_py.monolib.ShaderTextures.from_folder(str(folder))

            images = textures.global_textures()
            images = [(name, i) for (name, i) in images.items() if i is not None]

            png_images = xc3_model_py.decode_images_png(
                [i for (_, i) in images], not flip
            )

            shader_images = {}
            for (name, image), png in zip(images, png_images):
                # TODO: use the path as the name.
                image.name = name
                shader_images[name] = import_image(image, png, "", 0)

            return shader_images

    return None


def import_map_root(
    operator,
    root,
    root_collection: bpy.types.Collection,
    blender_images: list[bpy.types.Image],
    shader_images: Dict[str, bpy.types.Image],
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
                            shader_images,
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
                        y_up_to_z_up @ Matrix(transform) @ y_up_to_z_up.inverted()
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
    model_name,
    blender_images: list[bpy.types.Image],
    shader_images: Dict[str, bpy.types.Image],
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
            material = root.models.materials[mesh.material_index]

            if not import_all_meshes:
                if base_lods is not None and mesh.lod_item_index not in base_lods:
                    continue

                if "_outline" in material.name or "_speff_" in material.name:
                    continue

            # Many materials are for meshes that won't be loaded.
            # Lazy load materials to improve import times.
            material_name = f"{model_name}.{mesh.material_index}.{material.name}"
            blender_material = bpy.data.materials.get(material_name)
            if blender_material is None:
                blender_material = import_material(
                    material_name,
                    material,
                    blender_images,
                    shader_images,
                    root.image_textures,
                    root.models.samplers,
                )

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

        match attribute.attribute_type:
            case xc3_model_py.vertex.AttributeType.TexCoord0:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord0", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord1:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord1", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord2:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord2", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord3:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord3", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord4:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord4", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord5:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord5", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord6:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord6", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord7:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord7", flip_uvs)
            case xc3_model_py.vertex.AttributeType.TexCoord8:
                import_uvs(operator, blender_mesh, indices, data, "TexCoord8", flip_uvs)
            case xc3_model_py.vertex.AttributeType.VertexColor:
                import_colors(blender_mesh, data, "VertexColor")
            case xc3_model_py.vertex.AttributeType.Blend:
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

    # The validate call may modify and reindex geometry.
    # Assign normals now that the mesh has been updated.
    for attribute in vertex_buffer.attributes:
        if attribute.attribute_type in [
            xc3_model_py.vertex.AttributeType.Normal,
            xc3_model_py.vertex.AttributeType.Normal2,
        ]:
            # Store the data to use with shader nodes.
            data = attribute.data[min_index : max_index + 1]
            import_colors(blender_mesh, data, "VertexNormal")

    for attribute in vertex_buffer.morph_blend_target:
        if attribute.attribute_type == xc3_model_py.vertex.AttributeType.Normal4:
            # Store the data to use with shader nodes.
            data = attribute.data[min_index : max_index + 1]
            import_colors(blender_mesh, data, "VertexNormal")

    blender_mesh.update()
    blender_mesh.validate()

    # Calculating normals for invalid meshes seems to cause inconsistent crashes.
    # Setting normals after updating and validating is more reliable on some Blender versions.
    if attribute := blender_mesh.color_attributes.get("VertexNormal"):
        normals = np.zeros(len(blender_mesh.vertices) * 4)
        attribute.data.foreach_get("color", normals)
        # We can't assume that the attribute data is normalized.
        normals = normalize(normals.reshape((-1, 4))[:, :3])
        blender_mesh.normals_split_custom_set_from_vertices(normals)

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
        modifier.thickness_vertex_group = 0.01

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
    # The in game data is technically only 8 bits per channel.
    # Use full precision to avoid rounding errors on export.
    attribute = blender_mesh.color_attributes.new(
        name=name, type="FLOAT_COLOR", domain="POINT"
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
