import bpy
import time
import logging
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
        options={'HIDDEN'},
        maxlen=255,
    )

    original_wimdo: StringProperty(
        name="Original Wimdo", description="The original .wimdo file to use to generate the new model")

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        export_wimdo(self, context, self.filepath,
                     self.original_wimdo.strip('\"'))
        return {'FINISHED'}


def export_wimdo(operator: bpy.types.Operator, context: bpy.types.Context, output_wimdo_path: str, wimdo_path: str):
    start = time.time()

    if context.object is None or not isinstance(context.object.data, bpy.types.Armature):
        operator.report({'ERROR'}, 'No armature selected')
        return

    # TODO: Create this from scratch eventually?
    root = xc3_model_py.load_model(wimdo_path, None)
    root.buffers.vertex_buffers = []
    root.buffers.index_buffers = []
    root.models.models[0].meshes = []

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
