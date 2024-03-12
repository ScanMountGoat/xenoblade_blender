import bpy
import time
import os
import logging
import math

from . import xc3_model_py
from .import_root import get_database_path, import_root, import_images
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, EnumProperty
from mathutils import Matrix


class ImportWismhd(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade map"""
    bl_idname = "import_scene.wismhd"
    bl_label = "Import Wismhd"

    filename_ext = ".wismhd"

    filter_glob: StringProperty(
        default="*.wismhd",
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
        import_wismhd(context, self.filepath, database_path)
        return {'FINISHED'}


def import_wismhd(context: bpy.types.Context, path: str, database_path: str):
    start = time.time()

    roots = xc3_model_py.load_map(path, database_path)

    end = time.time()
    print(f'Load {len(roots)} Roots: {end - start}')

    start = time.time()

    model_name = os.path.basename(path)
    for root in roots:
        blender_images = import_images(root)

        # Create an empty by setting the data to None.
        # Maps have no skeletons.
        root_obj = bpy.data.objects.new(model_name, None)
        # Convert from Y up to Z up.
        root_obj.matrix_world = Matrix.Rotation(math.radians(90), 4, 'X')
        bpy.context.collection.objects.link(root_obj)

        import_root(root, blender_images, root_obj)

    end = time.time()
    print(f'Import Blender Scene: {end - start}')
