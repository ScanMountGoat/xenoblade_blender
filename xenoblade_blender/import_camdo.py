from pathlib import Path
import bpy
import time
import os
import logging
import math

from .import_root import get_image_folder, import_armature, import_model_root, import_images

from . import xc3_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, CollectionProperty
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

    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={'HIDDEN', 'SKIP_SAVE'})

    pack_images: BoolProperty(
        name="Pack Images",
        description="Pack all images into the Blender file. Increases memory usage and import times but makes the Blender file easier to share by not creating additional files",
        default=True
    )

    # TODO: Only show this if packed is unchecked
    image_folder: StringProperty(
        name="Image Folder", description="The folder for the imported images. Defaults to the file's parent folder if not set")

    def execute(self, context: bpy.types.Context):
        # Log any errors from Rust.
        log_fmt = '%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

        image_folder = get_image_folder(self.image_folder, self.filepath)

        # TODO: merge armatures?
        folder = Path(self.filepath).parent
        for file in self.files:
            abs_path = str(folder.joinpath(file.name))
            import_camdo(context, abs_path, self.pack_images, image_folder)

        return {'FINISHED'}


def import_camdo(context: bpy.types.Context, path: str, pack_images: bool, image_folder: str):
    start = time.time()

    root = xc3_model_py.load_model_legacy(path)

    end = time.time()
    print(f"Load Root: {end - start}")

    start = time.time()

    model_name = os.path.basename(path)
    blender_images = import_images(
        root, model_name, pack_images, image_folder, flip=False)
    armature = import_armature(context, root, model_name)
    import_model_root(root, blender_images, armature, flip_uvs=False)

    # Convert from Y up to Z up.
    armature.matrix_world = Matrix.Rotation(math.radians(90), 4, 'X')

    end = time.time()
    print(f"Import Blender Scene: {end - start}")
