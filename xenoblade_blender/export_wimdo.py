import bpy
import time
import logging
import numpy as np
from pathlib import Path

from .export_root import export_mesh

from . import xc3_model_py

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty


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

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        export_wimdo(self, context, self.filepath, self.original_wimdo.strip('"'))
        return {"FINISHED"}


def export_wimdo(
    operator: bpy.types.Operator,
    context: bpy.types.Context,
    output_wimdo_path: str,
    wimdo_path: str,
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

    for object in armature.children:
        export_mesh(root, object, combined_weights, original_meshes, morph_names)

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
