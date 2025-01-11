from pathlib import Path
import bpy
import time
import os

from .import_root import (
    get_database_path,
    get_image_folder,
    import_armature,
    import_model_root,
    import_images,
    import_monolib_shader_images,
    init_logging,
)

from . import xc3_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, CollectionProperty


class ImportWimdo(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade Switch model"""

    bl_idname = "import_scene.wimdo"
    bl_label = "Import Wimdo"

    filename_ext = ".wimdo"

    filter_glob: StringProperty(
        default="*.wimdo",
        options={"HIDDEN"},
        maxlen=255,
    )

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"}
    )

    pack_images: BoolProperty(
        name="Pack Images",
        description="Pack all images into the Blender file. Increases memory usage and import times but makes the Blender file easier to share by not creating additional files",
        default=True,
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

    import_outlines: BoolProperty(
        name="Import Outlines",
        description="Import data required to render and export outline meshes",
        default=True,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        database_path = get_database_path()
        image_folder = get_image_folder(self.image_folder, self.filepath)

        # TODO: merge armatures?
        folder = Path(self.filepath).parent
        for file in self.files:
            abs_path = str(folder.joinpath(file.name))
            self.import_wimdo(
                context,
                abs_path,
                database_path,
                self.pack_images,
                image_folder,
                self.import_all_meshes,
                self.import_outlines,
            )

        return {"FINISHED"}

    def import_wimdo(
        self,
        context: bpy.types.Context,
        path: str,
        database_path: str,
        pack_images: bool,
        image_folder: str,
        import_all_meshes: bool,
        import_outlines: bool,
    ):
        start = time.time()

        database = xc3_model_py.shader_database.ShaderDatabase.from_file(database_path)
        root = xc3_model_py.load_model(path, database)

        shader_images = import_monolib_shader_images(path, flip=True)

        end = time.time()
        print(f"Load Root: {end - start}")

        start = time.time()

        model_name = os.path.basename(path)
        name = model_name.replace(".wimdo", "")
        blender_images = import_images(
            root, name, pack_images, image_folder, flip=True
        )
        armature = import_armature(self, context, root, model_name)
        import_model_root(
            self,
            root,
            name,
            blender_images,
            shader_images,
            armature,
            import_all_meshes,
            import_outlines,
            flip_uvs=True,
        )

        # Store the path to make exporting easier later.
        armature["original_wimdo"] = path

        end = time.time()
        print(f"Import Blender Scene: {end - start}")
