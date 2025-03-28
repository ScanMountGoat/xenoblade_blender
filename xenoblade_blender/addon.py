import bpy

from . import import_mot
from . import import_wimdo
from . import import_wismhd
from . import import_camdo
from . import export_wimdo
from . import import_idcm


def menu_import_mot(self, context):
    text = "Xenoblade Animation (.mot/.anm/.motstm_data)"
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


def menu_import_idcm(self, context):
    text = "Xenoblade Collisions (.idcm/.wiidcm)"
    self.layout.operator(import_idcm.ImportIdcm.bl_idname, text=text)


classes = [
    import_mot.ImportMot,
    import_wimdo.ImportWimdo,
    import_wismhd.ImportWismhd,
    import_camdo.ImportCamdo,
    export_wimdo.ExportWimdo,
    import_idcm.ImportIdcm,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_import_mot)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_wimdo)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_wismhd)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_camdo)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_idcm)

    bpy.types.TOPBAR_MT_file_export.append(menu_export_wimdo)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_import_mot)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_wimdo)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_wismhd)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_camdo)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_idcm)

    bpy.types.TOPBAR_MT_file_export.remove(menu_export_wimdo)
