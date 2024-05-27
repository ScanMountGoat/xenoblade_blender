# xenoblade_blender Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
