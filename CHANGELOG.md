# xenoblade_blender Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## unreleased
### Changed
* Optimized import times for models and animations.
* Improved the accuracy of the generated Z coordinate for normal maps.

### Fixed
* Fixed an incorrect description for the Game Version import option.
* Fixed an issue where alpha channels from textures were not properly assigned in materials.
* Fixed an issue where material assignments did not use texture useage hints as a fallback for missing database entries.
* Fixed an issue where model space anims did not import with correct transforms.
* Fixed an issue where models did not import with accurate smooth normals.
* Fixed an issue where models did not import with all vertex color and texture coordinate attributes.

## 0.1.0 - 2024-03-13
First public release!
