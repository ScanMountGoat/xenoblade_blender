# xenoblade_blender Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## unreleased
### Changed
* Improved accuracy of material assignments and reduced shader node count in some cases.

### Fixed
* Fixed an issue preventing the import of materials that use the same texture with multiple UVs in some cases.

## 0.20.0 - 2025-06-10
### Changed
* Improved accuracy of material assignments.
* Changed material import to automatically arrange shader nodes.
* Adjusted wimdo import and export to preserve the 4th component from the "VertexNormal" attribute used for normal map intensity for some models.

### Fixed
* Fixed an issue where material nodes for base color did not correctly handle gamma.

## 0.19.0 - 2025-03-28
### Added
* Added support for importing Xenoblade Chronicles X Definitive Edition wimdo models.
* Added support for importing animations with the `.anm` or `.motstm_data` extensions.

### Changed
* Improved accuracy of material assignments.
* Adjusted importing to only warn about missing `monolib/shader` folder if the folder does not exist.

## 0.18.7 - 2025-03-13
### Fixed
* Fixed an issue where skeletons would import with missing bones or unparented bones in some cases.

## 0.18.6 - 2025-03-05
### Fixed
* Fixed an issue where some wimdo and wismhd models would fail to import.

## 0.18.5 - 2025-03-05
### Changed
* Updated provided shader database file to include Xenoblade 3 update models.

### Fixed
* Fixed an issue where some wimdo models would export with invalid vertex normals.
* Fixed an issue where wimdo export would not correctly preserve additional vertex data for some models.
* Fixed an issue where wimdo import and export would use the incorrect texture sampler in some cases.
* Fixed an issue where small textures for wimdo exports would load invalid mipmap data in game in some cases.

## 0.18.4 - 2025-02-25
### Fixed
* Fixed an issue where some wimdo models would not rebuild all data correctly on export.

## 0.18.3 - 2025-02-21
### Changed
* Changed generated materials to not connect toon ramp nodes if gradient texture is missing to avoid dark models.

### Fixed
* Fixed an issue where animations would incorrectly apply scale in some cases.
* Fixed an issue where animations would import with incorrect root motion.

## 0.18.2 - 2025-01-23
### Changed
* Optimized texture file sizes for wimdo model exports in some cases.

### Fixed
* Fixed an issue where some wimdo model textures would export with invalid low resolution data.

## 0.18.1 - 2025-01-16
### Changed
* Optimized import times for animations.

### Fixed
* Fixed an issue where merging armatures would not always correctly merge all bones.

## 0.18.0 - 2025-01-14
### Added
* Added an option to merge wimdo and camdo armatures on import. Leave this unchecked for editing and exporting models.

### Changed
* Changed imported material names to use the model name as a prefix to avoid name conflicts across imports.

### Fixed
* Fixed an issue where some wimdo models would import with empty or incomplete armatures.

## 0.17.0 - 2024-12-20
### Added
* Added support for importing collisions from .wiidcm or .idcm files.

### Fixed
* Fixed an issue where some wismhd maps would import with incorrect texture assignments.

## 0.16.3 - 2024-12-13
### Fixed
* Fixed an issue where some wimdo bones would import with incorrect transforms.
* Fixed an issue where some bones would export with incorrect transforms for wimdo export.

## 0.16.2 - 2024-12-11
### Fixed
* Fixed an issue where wimdo models would export with incorrect bone parenting information in some cases.

## 0.16.1 - 2024-12-03
### Added
* Added support for root motion to mot animation loading.

### Removed
* Removed the "Game Version" property for wimdo and wismhd import. Shader database entries are now selected automatically based on the file data.

## 0.16.0 - 2024-11-27
### Added
* Added support for exporting vertex groups for bones in the skeleton but not in the wimdo bone list.

### Changed
* Adjusted material node creation to produce fewer nodes in most cases by lazily creating UV nodes.
* Improved accuracy of material assignments.

### Fixed
* Fixed an issue where importing would fail for materials with invalid texture indices.
* Fixed an issue where some global textures would not import for some models.

## 0.15.4 - 2024-11-19
### Fixed
* Fixed an issue where editing a new material would also edit the original material in some cases.
* Fixed a compatibility issue preventing import in Blender 4.3.

## 0.15.3 - 2024-11-15
### Changed
* Improved accuracy of material assignments.
* Optimized import times for images.
* Improved the error message when exporting a model with vertex group names not supported by the original wimdo.
* Improved the error message when importing a camdo model with a missing casmt file.

## 0.15.2 - 2024-10-25
### Added
* Added support for parameter layering to generated materials.

### Changed
* Organized material nodes into frames.

### Fixed
* Fixed an issue where some materials would render as black due to incorrect blending settings.
* Fixed an issue where generated materials for camdo models would not use sampler information.
* Fixed an issue where skin weights would not export correctly for meshes with outlines.

## 0.15.1 - 2024-10-18
### Changed
* Improved node layout for materials with multiple texture layers.
* Improved the error message when enabling the addon on an unsupported Blender version.
* Adjusted importing to produce a warning if the armature cannot be created.

### Fixed
* Fixed an issue where some hair materials did not correctly generate toon gradient nodes.

## 0.15.0 - 2024-10-11
### Added
* Added support for toon gradient textures to generated materials.
* Added support for editing the selected toon gradient ramp for wimdo exports.

### Fixed
* Fixed an issue where some Xenoblade 2 models did not use the correct color layering information.
* Fixed an issue where global textures would not import correctly for some models.

## 0.14.2 - 2024-09-25
### Added
* Added support for textures in the monolib/shader folder to generated materials.

### Changed
* Improved precision for vertex color import.

### Fixed
* Fixed an issue where exporting wimdo models after removing all meshes with shape keys could cause crashes in game.
* Fixed an issue where texture layers would only assign the first channel.

## 0.14.1 - 2024-09-18
### Added
* Added support for color map layering to generated materials.
* Added support for adding new image textures for wimdo exports.

### Changed
* Improved color texture assignment accuracy for generated materials.
* Improved ambient occlusion texture assignment accuracy for generated materials.

## 0.14.0 - 2024-08-30
### Added
* Added support for normal map layering to generated materials.

### Changed
* Improved appearance of textures in game for wimdo model exports when high quality textures have not been streamed in yet.

### Fixed
* Fixed an issue where exporting would fail if an image texture node did not have the correct label.

## 0.13.0 - 2024-08-16
### Added
* Added an option to replaces textures with assigned images from current scene for wimdo export.
* Added an option to specify an image folder for replacing textures for wimdo export.

### Changed
* Adjusted wimdo export to report warnings if vertex groups are not part of the original skeleton.
* Improved color texture assignment accuracy for generated materials.
* Improved texture alpha assignment accuracy for generated materials.
* Improved accuracy of vertex data rebuilding.

### Fixed
* Fixed an issue where packed images would import without alpha channels.

## 0.12.0 - 2024-08-02
### Added
* Added support for multiplicative blending to generated materials.

### Changed
* Changed solidify modifier to use the "OutlineThickness" vertex group for thickness.
* Changed wimdo export to use the "OutlineThickness" vertex group for "OutlineVertexColor" alpha.
* Adjusted wimdo export to report warnings if outlines cannot be generated.
* Improved accuracy of vertex data rebuilding.

### Fixed
* Fixed an issue where some camdo models would not correctly remap material texture assignments.
* Fixed an issue where importing would fail if Pack Images was not checked.

## 0.11.1 - 2024-07-29
### Changed
* Improved the error message when attempting to install the addon on an unsupported Blender version.

### Fixed
* Fixed an issue where exporting would fail if an image texture did not use the expected naming convention.
* Fixed an issue where some models would not correctly generate outline meshes for all relevant meshes.

## 0.11.0 - 2024-07-26
### Added
* Added support for outline rendering using the solidify modifier.
* Added support for exporting outline data for wimdo exports.

### Changed
* Improved material assignment accuracy for Xenoblade X camdo models using a shader database.

### Fixed
* Fixed an issue where models would fail to import if they exceeded Blender's limit of 8 UVs. UVs past the limit will be skipped.
* Fixed an issue where some wimdo models would not export properly due to incorrectly saving skeleton data.

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
