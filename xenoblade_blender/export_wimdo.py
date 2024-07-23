import bpy
import time
import logging
import numpy as np
from pathlib import Path
import re

from .export_root import ExportException, copy_material, export_mesh

from . import xc3_model_py

from bpy_extras.io_utils import ExportHelper
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

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        try:
            export_wimdo(
                self,
                context,
                self.filepath,
                self.original_wimdo.strip('"'),
                self.create_speff_meshes,
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
    root.buffers.index_buffers = []
    root.models.models[0].meshes = []

    # Initialize weight buffer to share with all meshes.
    # TODO: Missing bone names?
    bone_names = root.buffers.weights.weight_buffer(16385).bone_names
    combined_weights = xc3_model_py.skinning.SkinWeights(
        np.array([]), np.array([]), bone_names
    )

    # Use a consistent ordering since Blender collections don't have one.
    sorted_objects = [o for o in armature.children]
    sorted_objects.sort(key=lambda o: name_sort_index(o.name))
    for object in sorted_objects:
        export_mesh(
            context,
            root,
            object,
            combined_weights,
            original_meshes,
            original_materials,
            morph_names,
            create_speff_meshes,
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
