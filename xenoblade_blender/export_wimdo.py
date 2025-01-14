import bpy
import time
import numpy as np
from pathlib import Path
import re
import os

from xenoblade_blender.import_root import init_logging

from .export_root import (
    ExportException,
    copy_material,
    export_mesh,
    image_index_to_replace,
)

from . import xc3_model_py

from bpy_extras.io_utils import ExportHelper
from bpy_extras import image_utils
from bpy.props import StringProperty, BoolProperty


class ExportWimdo(bpy.types.Operator, ExportHelper):
    """Export a Xenoblade Switch model"""

    bl_idname = "export_scene.wimdo"
    bl_label = "Export Wimdo"

    filename_ext = ".wimdo"

    filter_glob: StringProperty(
        default="*.wimdo",
        options={"HIDDEN"},
        maxlen=255,
    )

    original_wimdo: StringProperty(
        name="Original Wimdo",
        description="The original .wimdo file to use to generate the new model. Defaults to the armature's original_wimdo custom property if not set",
    )

    create_speff_meshes: BoolProperty(
        name="Create _speff meshes",
        description="Create additional copies of meshes with _speff materials. This only affects Xenoblade 3",
        default=True,
    )

    export_images: BoolProperty(
        name="Export Images",
        description="Replace images in the exported wimdo from the Blender scene",
        default=False,
    )

    # TODO: Only show this if export images is checked
    image_folder: StringProperty(
        name="Image Folder",
        description="Use images from this folder instead of the Blender scene. Has no effect if Export Images is unchecked",
        default="",
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        try:
            export_wimdo(
                self,
                context,
                self.filepath,
                self.original_wimdo.strip('"'),
                self.create_speff_meshes,
                self.export_images,
                self.image_folder,
            )
        except ExportException as e:
            self.report({"ERROR"}, str(e))
            return {"FINISHED"}

        return {"FINISHED"}


def name_sort_index(name: str):
    # Use integer sorting for any chunks of chars that are integers.
    # This avoids unwanted behavior of alphabetical sorting like "10" coming before "2".
    return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", name)]


def export_wimdo(
    operator: bpy.types.Operator,
    context: bpy.types.Context,
    output_wimdo_path: str,
    wimdo_path: str,
    create_speff_meshes: bool,
    export_images: bool,
    image_folder: str,
):
    start = time.time()

    armature = context.object
    if armature is None or not isinstance(armature.data, bpy.types.Armature):
        operator.report({"ERROR"}, "No armature selected")
        return

    # The path specified in export settings should have priority.
    if wimdo_path == "":
        wimdo_path = armature.get("original_wimdo", "")

    # TODO: Create this from scratch eventually?
    root = xc3_model_py.load_model(wimdo_path, None)
    original_meshes = [m for m in root.models.models[0].meshes]
    original_materials = [copy_material(m) for m in root.models.materials]
    morph_names = root.models.morph_controller_names
    root.buffers.vertex_buffers = []
    root.buffers.outline_buffers = []
    root.buffers.index_buffers = []
    root.models.models[0].meshes = []

    # Initialize weight buffer to share with all meshes.
    # TODO: Missing bone names?
    bone_names = root.buffers.weights.weight_buffer(16385).bone_names

    # TODO: ignore groups without weights?
    vertex_group_names = set()
    for o in armature.children:
        for vg in o.vertex_groups:
            if vg.name != "OutlineThickness":
                vertex_group_names.add(vg.name)

    skinning = root.models.skinning
    if skinning is not None:
        for name in vertex_group_names:
            if name not in bone_names:
                # Bones need bounds to render in game.
                # TODO: Is it ok to not set constraints on added bones?
                bounds = xc3_model_py.skinning.BoneBounds(
                    [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], 0.1
                )
                bone = xc3_model_py.skinning.Bone(name, False, bounds, None)
                skinning.bones.append(bone)

                # The skinning bone list needs to match the weights.
                bone_names.append(name)

    combined_weights = xc3_model_py.skinning.SkinWeights(
        np.zeros((0, 4), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32), bone_names
    )

    image_replacements = set()

    # Use a consistent ordering since Blender collections don't have one.
    sorted_objects = [o for o in armature.children if o.type == "MESH"]
    sorted_objects.sort(key=lambda o: name_sort_index(o.name))
    for object in sorted_objects:
        export_mesh(
            context,
            operator,
            root,
            object,
            combined_weights,
            original_meshes,
            original_materials,
            morph_names,
            create_speff_meshes,
            image_replacements,
        )

    # The vertex shaders have a limited SSBO size for preskinned transform matrices.
    # XC3 has a higher limit (12000) than XC2, so use the lower limit (8000) to be safe.
    weight_count = len(combined_weights.bone_indices)
    if weight_count > 8000:
        message = f"Unique weight count {weight_count} exceeds in game limit of 8000."
        message += " Simplify or quantize weights or limit the number of vertices."
        raise ExportException(message)

    root.buffers.weights.update_weights(combined_weights)

    # Detect the default import behavior of a single lod level in each group.
    # Setting the lod count avoids needing to export LOD or speff meshes.
    lod_data = root.models.lod_data
    if lod_data is not None:
        base_lod_indices = [g.base_lod_index for g in lod_data.groups]
        if all(
            m.lod_item_index in base_lod_indices for m in root.models.models[0].meshes
        ):
            for g in lod_data.groups:
                g.lod_count = 1

    if export_images:
        # TODO: move this logic to xc3_model_py?
        if image_folder != "":
            export_external_images(root, image_folder)
        else:
            export_internal_images(root, image_replacements)

    end = time.time()
    print(f"Create ModelRoot: {end - start}")

    start = time.time()

    # TODO: Error if no wimdo path or path cannot be found
    mxmd = xc3_model_py.Mxmd.from_file(wimdo_path)
    msrd = xc3_model_py.Msrd.from_file(str(Path(wimdo_path).with_suffix(".wismt")))

    new_mxmd, new_msrd = root.to_mxmd_model(mxmd, msrd)

    new_mxmd.save(output_wimdo_path)
    new_msrd.save(str(Path(output_wimdo_path).with_suffix(".wismt")))

    end = time.time()
    print(f"Export Files: {end - start}")


def export_internal_images(root, image_replacements):
    # TODO: Support adding images
    # TODO: Check out of bounds indices.
    # TODO: Will this encode some images more than once?
    start = time.time()

    # Sort for proper handling of added indices.
    image_replacements = sorted(image_replacements, key=lambda x: x[0])

    validate_image_replacements(root, [i for i, _ in image_replacements])

    encode_image_args = internal_encode_image_args(root, image_replacements)

    end = time.time()
    print(f"Load images: {end - start}")

    # Encode images in parallel for better performance.
    image_textures = xc3_model_py.encode_images_rgbaf32(encode_image_args)

    for (i, _), image in zip(image_replacements, image_textures):
        if i < len(root.image_textures):
            root.image_textures[i] = image
        else:
            # We've already validated that new indices are contiguous.
            root.image_textures.append(image)


def internal_encode_image_args(root, image_replacements):
    encode_image_args = []
    for i, image in image_replacements:
        original_image = None
        if i < len(root.image_textures):
            original_image = root.image_textures[i]
        args = encode_args_from_image(image, original_image)
        encode_image_args.append(args)
    return encode_image_args


def validate_image_replacements(root, new_indices):
    # Validate that indices form a valid range for replacement.
    indices = set(range(len(root.image_textures)))
    for i in new_indices:
        indices.add(i)

    for i, index in enumerate(sorted(indices)):
        if i != index:
            message = f"Expected image index {i} but found {index}. Added textures should not skip indices."
            raise ExportException(message)


def encode_args_from_image(image, original_image):
    width, height = image.size
    # Flip vertically to match in game.
    # TODO: How to speed this up?
    image_data = np.zeros(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(image_data)

    image_data = np.flip(
        image_data.reshape((width, height, 4)),
        axis=0,
    )

    if original_image is not None:
        args = xc3_model_py.EncodeSurfaceRgba32FloatArgs(
            width,
            height,
            1,
            xc3_model_py.ViewDimension.D2,
            original_image.image_format,
            original_image.mipmap_count > 1,
            image_data.reshape(-1),
            original_image.name,
            original_image.usage,
        )
    else:
        args = xc3_model_py.EncodeSurfaceRgba32FloatArgs(
            width,
            height,
            1,
            xc3_model_py.ViewDimension.D2,
            xc3_model_py.ImageFormat.BC7Unorm,
            True,
            image_data.reshape(-1),
            "",
            xc3_model_py.material.TextureUsage.Col,
        )

    return args


def export_external_images(root, image_folder: str):
    image_indices_args = []
    dds_indices_images = []
    start = time.time()
    for name in os.listdir(image_folder):
        path = os.path.join(image_folder, name)

        i = image_index_to_replace(root.image_textures, name)
        if i is None:
            continue

        image_texture = None
        if i < len(root.image_textures):
            image_texture = root.image_textures[i]

        if path.endswith(".dds"):
            dds = xc3_model_py.Dds.from_file(path)
            if image_texture is not None:
                image = xc3_model_py.ImageTexture.from_dds(
                    dds, image_texture.name, image_texture.usage
                )
                dds_indices_images.append((i, image))
            else:
                image = xc3_model_py.ImageTexture.from_dds(
                    dds, "", xc3_model_py.material.TextureUsage.Col
                )
                dds_indices_images.append((i, image))
        else:
            # Assume other file types are images.
            image = image_utils.load_image(
                path, place_holder=True, check_existing=False
            )
            args = encode_args_from_image(image, image_texture)
            image_indices_args.append((i, args))

    end = time.time()
    print(f"Load images: {end - start}")

    new_indices = [i for i, _ in image_indices_args]
    new_indices.extend(i for i, _ in dds_indices_images)
    validate_image_replacements(root, new_indices)

    # Encode images in parallel for better performance.
    encode_image_args = [args for _, args in image_indices_args]
    image_textures = xc3_model_py.encode_images_rgbaf32(encode_image_args)
    image_indices = [i for i, _ in image_indices_args]

    # TODO: Should DDS take always priority over PNG to handle duplicate indices?
    image_replacements = set()
    for i, image in zip(image_indices, image_textures):
        image_replacements.add((i, image))

    for i, image in dds_indices_images:
        image_replacements.add((i, image))

    # Sort for proper handling of added indices.
    image_replacements = sorted(image_replacements, key=lambda x: x[0])

    print(image_replacements)

    # TODO: Avoid replacing a texture more than once.
    for i, image_texture in image_replacements:
        if i < len(root.image_textures):
            root.image_textures[i] = image_texture
        else:
            # We've already validated that new indices are contiguous.
            root.image_textures.append(image_texture)
