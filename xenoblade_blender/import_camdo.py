import bpy
import time
import os
import logging
import math

from .import_root import import_armature, import_root, import_images

from . import xc3_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from mathutils import Matrix


class ImportCamdo(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade Wii U model"""
    bl_idname = "import_scene.camdo"
    bl_label = "Import Camdo"

    filename_ext = ".camdo"

    filter_glob: StringProperty(
        default="*.camdo",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        import_camdo(context, self.filepath)
        return {'FINISHED'}


def import_camdo(context: bpy.types.Context, path: str):
    start = time.time()

    root = xc3_model_py.load_model_legacy(path)

    end = time.time()
    print(f"Load Root: {end - start}")

    start = time.time()

    model_name = os.path.basename(path)
    blender_images = import_images(root)
    armature = import_armature(context, root, model_name)
    import_root(root, blender_images, armature)

    # Convert from Y up to Z up.
    armature.matrix_world = Matrix.Rotation(math.radians(90), 4, 'X')

    end = time.time()
    print(f"Import Blender Scene: {end - start}")
