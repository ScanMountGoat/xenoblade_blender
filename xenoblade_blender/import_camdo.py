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
    init_logging,
    merge_armatures,
)

from . import xc3_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, CollectionProperty


class ImportCamdo(bpy.types.Operator, ImportHelper):
    """Import a Xenoblade Wii U model"""

    bl_idname = "import_scene.camdo"
    bl_label = "Import Camdo"

    filename_ext = ".camdo"

    filter_glob: StringProperty(
        default="*.camdo",
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

    merge_armatures: BoolProperty(
        name="Merge Armatures",
        description="Merge bones from all imported armatures and the selected armature.",
        default=False,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        database_path = get_database_path()
        database = xc3_model_py.shader_database.ShaderDatabase.from_file(database_path)

        image_folder = get_image_folder(self.image_folder, self.filepath)

        armatures = []

        folder = Path(self.filepath).parent
        for file in self.files:
            abs_path = str(folder.joinpath(file.name))
            armature = self.import_camdo(
                context, abs_path, database, self.pack_images, image_folder
            )
            armatures.append(armature)

        if self.merge_armatures:
            merge_armatures(self, context, armatures)

        return {"FINISHED"}

    def import_camdo(
        self,
        context: bpy.types.Context,
        path: str,
        database: xc3_model_py.shader_database.ShaderDatabase,
        pack_images: bool,
        image_folder: str,
    ) -> bpy.types.Object:
        start = time.time()

        root = xc3_model_py.load_model_legacy(path, database)

        end = time.time()
        print(f"Load Root: {end - start}")

        start = time.time()

        model_name = os.path.basename(path)
        name = model_name.replace(".camdo", "")

        blender_images = import_images(
            root,
            name,
            pack_images,
            image_folder,
            flip=False,
        )

        shader_images = {}

        armature = import_armature(self, context, root, model_name)

        import_model_root(
            self,
            root,
            name,
            blender_images,
            shader_images,
            armature,
            import_all_meshes=True,
            import_outlines=True,
            flip_uvs=False,
        )

        end = time.time()
        print(f"Import Blender Scene: {end - start}")

        return armature
