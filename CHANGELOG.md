# xenoblade_blender Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## unreleased
### Added
* Added an option to save unpacked PNG textures when importing to reduce memory usage and import times.

### Changed
* Optimized import times for models and maps with many materials and images.
* Adjusted images and UVs to have the expected vertical orientation.

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
