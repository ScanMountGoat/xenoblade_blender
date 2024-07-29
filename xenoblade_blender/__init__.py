import bpy

from . import import_mot
from . import import_wimdo
from . import import_wismhd
from . import import_camdo
from . import export_wimdo

bl_info = {
    "name": "Xenoblade Blender",
    "description": "Import and export Xenoblade models, maps, and animations",
    "author": "ScanMountGoat (SMG)",
    "version": (0, 11, 0),
    "blender": (4, 1, 0),
    "location": "File > Import-Export",
    "warning": "",
    "doc_url": "https://github.com/ScanMountGoat/xenoblade_blender/wiki",
    "tracker_url": "https://github.com/ScanMountGoat/xenoblade_blender/issues",
    "category": "Import-Export",
}


def menu_import_mot(self, context):
    text = "Xenoblade Animation (.mot)"
    self.layout.operator(import_mot.ImportMot.bl_idname, text=text)


def menu_import_wimdo(self, context):
    text = "Xenoblade Model (.wimdo)"
    self.layout.operator(import_wimdo.ImportWimdo.bl_idname, text=text)


def menu_export_wimdo(self, context):
    text = "Xenoblade Model (.wimdo)"
    self.layout.operator(export_wimdo.ExportWimdo.bl_idname, text=text)


def menu_import_wismhd(self, context):
    text = "Xenoblade Map (.wismhd)"
    self.layout.operator(import_wismhd.ImportWismhd.bl_idname, text=text)


def menu_import_camdo(self, context):
    text = "Xenoblade Model (.camdo)"
    self.layout.operator(import_camdo.ImportCamdo.bl_idname, text=text)


classes = [
    import_mot.ImportMot,
    import_wimdo.ImportWimdo,
    import_wismhd.ImportWismhd,
    import_camdo.ImportCamdo,
    export_wimdo.ExportWimdo,
]


def register():
    if bpy.app.version < bl_info["blender"]:
        current_version = ".".join(str(v) for v in bpy.app.version)
        expected_version = ".".join(str(v) for v in bl_info["blender"])
        raise ImportError(
            f"Blender version {current_version} is incompatible. Use version {expected_version} or later."
        )

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_import_mot)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_wimdo)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_wismhd)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_camdo)

    bpy.types.TOPBAR_MT_file_export.append(menu_export_wimdo)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_import_mot)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_wimdo)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_wismhd)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_camdo)

    bpy.types.TOPBAR_MT_file_export.remove(menu_export_wimdo)


if __name__ == "__main__":
    register()
