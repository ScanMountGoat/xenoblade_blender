# xenoblade_blender Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## unreleased
### Changed
* Improved material assignment accuracy for Xenoblade X camdo models using a shader database.

### Fixed
* Fixed an issue where models would fail to import if they exceeded Blender's limit of 8 UVs. UVs past the limit will be skipped.

## 0.10.0 - 2024-07-02
### Added
* Added support for specular color maps to generated materials.
* Added support for texture scale to generated materials.
* Added support for ambient occlusion maps to generated materials.
* Added support for vertex color to generated materials.
* Added support for UV map selection to generated materials.

### Fixed
* Fixed an issue where glossiness would use the incorrect material assignment.
* Fixed an issue where material edits would not apply to generated "_speff" materials for Xenoblade 3.

## 0.9.1 - 2024-06-24
### Fixed
* Fixed an issue where importing would fail if the UI language was not set to English.

## 0.9.0 - 2024-06-20
### Added
* Added support for adding new materials for wimdo exports.
* Added support for changing material texture assignments for wimdo exports.

### Changed
* Changed the import behavior for wismhd maps to use collection instances instead of linked duplicate meshes.
* Changed exception message formatting to include the inner errors for easier debugging.
* Moved normal map Z calculation to a node group.

## 0.8.2 - 2024-06-12
### Added
* Added validation for material names and index prefixes to wimdo export.

### Fixed
* Fixed an issue where shape keys would import with the wrong orientation.
* Fixed an issue where wimdo export error messages would use the wrong mesh name.
* Fixed an issue where shape keys would export with the wrong indices.
* Fixed an issue where camdo models would not correctly import all bones.
* Fixed an issue where some camdo models would not import due to texture loading errors.
* Fixed an issue where enabling speff mesh export would incorrectly generate additional meshes for Xenoblade 1 DE and Xenoblade 2.

## 0.8.1 - 2024-06-07
### Fixed
* Fixed an issue where temporary mesh processing did not work properly for wimdo export.

## 0.8.0 - 2024-06-07
### Added
* Added an option to generate meshes with "_speff" materials on export for Xenoblade 3.
* Added error reporting for errors preventing import or export operations from succeeding.

### Changed
* Improved the layout of nodes in generated materials.
* Changed the name of materials to include an index prefix to handle materials with duplicate names.
* Changed the label for image texture nodes to include the material texture index.
* Changed exporting behavior to automatically apply transforms and any non armature modifiers.
* Changed image names to always include the image index similar to xc3_tex.
* Changed armatures, models, and animations to use Blender's coordinate system on import.

### Fixed
* Fixed an issue where .camdo models would not import due to unsupported sampler data.
* Fixed an issue where models with more than one normal or UV per vertex would not export properly.
* Fixed an issue where animation import would sometimes use transforms from a previous animation.

## 0.7.0 - 2024-06-02
### Added
* Added the `original_wimdo` custom property to .wimdo armatures on import. This value will be saved with the Blender scene and avoids needing to set the original file path for exporting in most cases.

### Fixed
* Fixed an issue where wimdo export would not correctly detect mesh index prefixes.
* Fixed an issue where wimdo export would fail if a mesh did not have any shape keys.
* Fixed an issue where exported .wimdo models would not correctly render shadows, causing massive performance problems in game.
* Fixed an issue where materials would not use the correct UV wrap modes for image textures.
* Fixed an issue where some .wimdo files did not load the correct meshes for the base level of detail on import.
* Fixed an issue where non square textures would not import with the correct layout.

## 0.6.0 - 2024-05-27
### Added
* Added support for emissive maps to generated materials.
* Added support for skin weights to exported .wimdo files.
* Added support for additional UV and color attributes to exported .wimdo files.
* Added support for shape keys to exported .wimdo files.

### Fixed
* Fixed an issue where some model exports would be invisible in game. This mostly affected Xenoblade 3 models.

## 0.5.0 - 2024-05-02
### Added
* Added an option to import all meshes for .wismhd or .wimdo files.

### Changed
* Adjusted mesh names to include the mesh index to match the in game order in the outliner.
* Improved accuracy of base level of detail (LOD) detection.

### Fixed
* Fixed an issue where animations did not loop properly due to the final keyframe not extrapolating to the final frame.

## 0.4.0 - 2024-04-30
### Added
* Added an option to save unpacked PNG textures when importing to reduce memory usage and import times.

### Changed
* Optimized import times for models and maps with many materials and images.
* Adjusted images and UVs to have the expected vertical orientation.
* Adjusted .wimdo and .camdo import dialogs to support selecting multiple files.

### Fixed
* Fixed an issue where some animation files failed to load.

## 0.3.1 - 2024-04-19
### Fixed
* Fixed an issue where map files failed to load.

## 0.3.0 - 2024-04-19
### Added
* Added support for importing Xenoblade Chronicles X models from .camdo files.
* Added shape key support for model imports.
* Added experimental support for exporting .wimdo models under File > Export.

## 0.2.0 - 2024-04-01
### Changed
* Optimized import times for models and animations.
* Improved the accuracy of the generated Z coordinate for normal maps.
* Changed supported Blender version to 4.1.

### Fixed
* Fixed an incorrect description for the Game Version import option.
* Fixed an issue where alpha channels from textures were not properly assigned in materials.
* Fixed an issue where material assignments did not use texture useage hints as a fallback for missing database entries.
* Fixed an issue where model space anims did not import with correct transforms.
* Fixed an issue where models did not import with accurate smooth normals.
* Fixed an issue where models did not import with all vertex color and texture coordinate attributes.
* Fixed an issue where materials did not use the correct gamma settings for color vs non color textures.

## 0.1.0 - 2024-03-13
First public release!
