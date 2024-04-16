import bpy
import time
import os
import logging
import math
from pathlib import Path

from .import_root import get_database_path, import_armature, import_root, import_images
from .export_root import export_mesh

from . import xc3_model_py

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, EnumProperty
from mathutils import Matrix


class ExportWimdo(bpy.types.Operator, ExportHelper):
    """Export a Xenoblade Switch model"""
    bl_idname = "export_scene.wimdo"
    bl_label = "Export Wimdo"

    filename_ext = ".wimdo"

    filter_glob: StringProperty(
        default="*.wimdo",
        options={'HIDDEN'},
        maxlen=255,
    )

    original_wimdo: StringProperty(
        name="Original Wimdo", description="The original .wimdo file to use to generate the new model")

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        export_wimdo(context, self.filepath, self.original_wimdo.strip('\"'))
        return {'FINISHED'}


def export_wimdo(context: bpy.types.Context, output_wimdo_path: str, wimdo_path: str):
    # TODO: Error if no armature is selected?
    start = time.time()

    # TODO: Create this from scratch eventually?
    root = xc3_model_py.load_model(wimdo_path, None)
    root.groups[0].buffers[0].vertex_buffers = []
    root.groups[0].buffers[0].index_buffers = []
    root.groups[0].models[0].models[0].meshes = []

    # TODO: Don't assume an armature is selected?
    armature = context.object
    for object in armature.children:
        export_mesh(root, object)

    end = time.time()
    print(f"Create ModelRoot: {end - start}")

    start = time.time()

    # TODO: Error if no wimdo path or path cannot be found
    mxmd = xc3_model_py.Mxmd.from_file(wimdo_path)
    msrd = xc3_model_py.Msrd.from_file(
        str(Path(wimdo_path).with_suffix('.wismt')))

    new_mxmd, new_msrd = root.to_mxmd_model(mxmd, msrd)

    new_mxmd.save(output_wimdo_path)
    new_msrd.save(str(Path(output_wimdo_path).with_suffix('.wismt')))

    end = time.time()
    print(f"Export Files: {end - start}")
