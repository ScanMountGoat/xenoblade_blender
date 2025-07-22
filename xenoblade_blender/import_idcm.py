from pathlib import Path
import bpy
import time
import numpy as np
import math

from xenoblade_blender.import_root import init_logging

from . import xc3_model_py
from mathutils import Matrix

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, CollectionProperty


class ImportIdcm(bpy.types.Operator, ImportHelper):
    """Import Xenoblade collision data"""

    bl_idname = "import_scene.idcm"
    bl_label = "Import Idcm"

    filename_ext = ".idcm"

    filter_glob: StringProperty(
        default="*.idcm;*.wiidcm",
        options={"HIDDEN"},
        maxlen=255,
    )

    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"}
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        folder = Path(self.filepath).parent
        for file in self.files:
            abs_path = str(folder.joinpath(file.name))
            self.import_idcm(context, abs_path)

        return {"FINISHED"}

    def import_idcm(
        self,
        context: bpy.types.Context,
        path: str,
    ):
        start = time.time()

        collisions = xc3_model_py.load_collisions(path)

        end = time.time()
        print(f"Load Collision: {end - start}")

        start = time.time()

        # Convert from Y up to Z up.
        y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")

        # TODO: Put these in a collection?
        for mesh in collisions.meshes:
            blender_mesh = bpy.data.meshes.new(mesh.name)

            # TODO: shared function in import_root?
            min_index = 0
            max_index = 0
            if mesh.indices.size > 0:
                min_index = mesh.indices.min()
                max_index = mesh.indices.max()

            indices = mesh.indices.astype(np.uint32) - min_index
            loop_start = np.arange(0, indices.shape[0], 3, dtype=np.uint32)
            loop_total = np.full(loop_start.shape[0], 3, dtype=np.uint32)

            blender_mesh.loops.add(indices.shape[0])
            blender_mesh.loops.foreach_set("vertex_index", indices)

            blender_mesh.polygons.add(loop_start.shape[0])
            blender_mesh.polygons.foreach_set("loop_start", loop_start)
            blender_mesh.polygons.foreach_set("loop_total", loop_total)

            positions = collisions.vertices[min_index : max_index + 1, :3]
            blender_mesh.vertices.add(positions.shape[0])
            blender_mesh.vertices.foreach_set("co", positions.reshape(-1))

            blender_mesh.update()
            blender_mesh.validate()

            # Convert from Y up to Z up.
            blender_mesh.transform(y_up_to_z_up)

            if len(mesh.instances) == 0:
                obj = bpy.data.objects.new(blender_mesh.name, blender_mesh)
                bpy.context.collection.objects.link(obj)
            else:
                for instance in mesh.instances:
                    obj = bpy.data.objects.new(blender_mesh.name, blender_mesh)
                    bpy.context.collection.objects.link(obj)

                    # Transform the instance using the in game coordinate system and convert back.
                    obj.matrix_world = (
                        y_up_to_z_up @ Matrix(instance) @ y_up_to_z_up.inverted()
                    )

        end = time.time()
        print(f"Import Blender Scene: {end - start}")
