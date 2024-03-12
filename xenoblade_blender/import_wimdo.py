import bpy
import time
import os
import logging
import math

from .import_root import get_database_path, import_armature, import_root, import_images

from . import xc3_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, EnumProperty
from mathutils import Matrix


class ImportWimdo(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade model"""
    bl_idname = "import_scene.wimdo"
    bl_label = "Import Wimdo"

    filename_ext = ".wimdo"

    filter_glob: StringProperty(
        default="*.wimdo",
        options={'HIDDEN'},
        maxlen=255,
    )

    game_version: EnumProperty(
        name="Game Version",
        description="Choose between two items",
        items=(
            ('XC1', "Xenoblade 1 DE", "Xenoblade Chronicles 1 Definitive Edition"),
            ('XC2', "Xenoblade 2", "Xenoblade Chronicles 2"),
            ('XC3', "Xenoblade 3", "Xenoblade Chronicles 3"),
        ),
        default='XC3',
    )

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        database_path = get_database_path(self.game_version)
        import_wimdo(context, self.filepath, database_path)
        return {'FINISHED'}


def import_wimdo(context: bpy.types.Context, path: str, database_path: str):
    start = time.time()

    root = xc3_model_py.load_model(path, database_path)

    end = time.time()
    print(f'Load Root: {end - start}')

    start = time.time()

    # TODO: Create a module for code shared with import_wismhd
    model_name = os.path.basename(path)
    blender_images = import_images(root)
    armature = import_armature(context, root, model_name)
    import_root(root, blender_images, armature)

    # Convert from Y up to Z up.
    armature.matrix_world = Matrix.Rotation(math.radians(90), 4, 'X')

    end = time.time()
    print(f'Import Blender Scene: {end - start}')
