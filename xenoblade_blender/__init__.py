import bpy

bl_info = {
    "name": "Xenoblade Blender",
    "description": "Import and export Xenoblade models, maps, and animations",
    "author": "ScanMountGoat (SMG)",
    "version": (0, 18, 3),
    "blender": (4, 1, 0),
    "location": "File > Import-Export",
    "warning": "",
    "doc_url": "https://github.com/ScanMountGoat/xenoblade_blender/wiki",
    "tracker_url": "https://github.com/ScanMountGoat/xenoblade_blender/issues",
    "category": "Import-Export",
}


def register():
    # Check the Blender version before importing the xc3_model_py module.
    # This avoids showing a DLL import error if the Python version
    # has changed between the current and expected Blender version.
    if bpy.app.version < bl_info["blender"]:
        current_version = ".".join(str(v) for v in bpy.app.version)
        expected_version = ".".join(str(v) for v in bl_info["blender"])
        raise ImportError(
            f"Blender version {current_version} is incompatible. Use version {expected_version} or later."
        )

    from . import addon

    addon.register()


def unregister():
    from . import addon

    addon.unregister()


if __name__ == "__main__":
    register()
