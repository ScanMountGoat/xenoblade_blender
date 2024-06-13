import bpy
import time
import os
import logging
import math

from . import xc3_model_py
from .import_root import (
    get_database_path,
    get_image_folder,
    import_map_root,
    import_images,
)
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, EnumProperty, BoolProperty
from mathutils import Matrix


class ImportWismhd(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade map"""

    bl_idname = "import_scene.wismhd"
    bl_label = "Import Wismhd"

    filename_ext = ".wismhd"

    filter_glob: StringProperty(
        default="*.wismhd",
        options={"HIDDEN"},
        maxlen=255,
    )

    game_version: EnumProperty(
        name="Game Version",
        description="The game version for the shader database",
        items=(
            ("XC1", "Xenoblade 1 DE", "Xenoblade Chronicles 1 Definitive Edition"),
            ("XC2", "Xenoblade 2", "Xenoblade Chronicles 2"),
            ("XC3", "Xenoblade 3", "Xenoblade Chronicles 3"),
        ),
        default="XC3",
    )

    pack_images: BoolProperty(
        name="Pack Images",
        description="Pack all images into the Blender file. Increases memory usage and import times but makes the Blender file easier to share by not creating additional files",
        default=False,
    )

    # TODO: Only show this if packed is unchecked
    image_folder: StringProperty(
        name="Image Folder",
        description="The folder for the imported images. Defaults to the file's parent folder if not set",
    )

    import_all_meshes: BoolProperty(
        name="Import All Meshes",
        description="Import all meshes regardless of LOD or material name",
        default=False,
    )

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        database_path = get_database_path(self.game_version)
        image_folder = get_image_folder(self.image_folder, self.filepath)

        import_wismhd(
            context,
            self.filepath,
            database_path,
            self.pack_images,
            image_folder,
            self.import_all_meshes,
        )
        return {"FINISHED"}


def import_wismhd(
    context: bpy.types.Context,
    path: str,
    database_path: str,
    pack_images: bool,
    image_folder: str,
    import_all_meshes: bool,
):
    start = time.time()

    roots = xc3_model_py.load_map(path, database_path)

    end = time.time()
    print(f"Load {len(roots)} Roots: {end - start}")

    start = time.time()

    model_name = os.path.basename(path)

    for i, root in enumerate(roots):
        name = model_name.replace(".wismhd", "")
        blender_images = import_images(
            root, f"{name}.root{i}", pack_images, image_folder, flip=True
        )

        # Maps have no skeletons.
        root_collection = bpy.data.collections.new(f"{name}.{i}")
        bpy.context.scene.collection.children.link(root_collection)

        import_map_root(
            root, root_collection, blender_images, import_all_meshes, flip_uvs=True
        )

    end = time.time()
    print(f"Import Blender Scene: {end - start}")
