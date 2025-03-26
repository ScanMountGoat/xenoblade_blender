# xenoblade_blender [![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ScanMountGoat/xenoblade_blender?include_prereleases)](https://github.com/ScanMountGoat/xenoblade_blender/releases/latest) [![wiki](https://img.shields.io/badge/wiki-guide-success)](https://github.com/scanmountgoat/xenoblade_blender/wiki)
A Blender addon for importing models, maps, and animations for Xenoblade Chronicles X, Xenoblade Chronicles 1 DE, Xenoblade Chronicles 2, and Xenoblade Chronicles 3. 

Report bugs or request new features in [issues](https://github.com/ScanMountGoat/xenoblade_blender/issues). Download the latest version from [releases](https://github.com/ScanMountGoat/xenoblade_blender/releases). Check the [wiki](https://github.com/ScanMountGoat/xenoblade_blender/wiki) for more usage information.

## Features
* Faster and more accurate than intermediate formats like FBX or glTF
* Import characters, objects, enemies, and weapon models from .wimdo and .camdo files
* Import maps from .wismhd files
* Import character animations from .mot files
* Import collision meshes from .wiidcm or .idcm files
* Decoded and embedded RGBA textures for easier use in Blender
* Material parameters, textures, and texture channel assignments based on decompiled in game shader code
* Export .wimdo models using an existing model as a base. See the [wiki](https://github.com/ScanMountGoat/xenoblade_blender/wiki/Export) for details.

## Getting Started
* Download the latest version of the addon supported by your Blender version from [releases](https://github.com/ScanMountGoat/xenoblade_blender/releases).
* Install the .zip file in Blender using Edit > Preferences > Addons > Install...
* Enable the addon if it is not already enabled.
* Extract the files from the arh and ard from your romfs dump of the game using [XbTool](https://github.com/AlexCSDev/XbTool/releases).
* Import any of the supported file types using the new menu options under File > Import.  

## Updating
Update the addon by reinstalling the latest version from [releases](https://github.com/ScanMountGoat/xenoblade_blender/releases). MacOS and Linux users can update without any additional steps.

> [!IMPORTANT]
> Windows users need to disable the addon, restart Blender, remove the addon, and install the new version to update.

## Building
Clone the repository with `git clone https://github.com/ScanMountGoat/xenoblade_blender --recursive`. 

xenoblade_blender uses [xc3_model_py](https://github.com/ScanMountGoat/xc3_model_py) for simplifying the addon code and achieving better performance than pure Python addons. 

xc3_model_py must be compiled from source after [installing Rust](https://www.rust-lang.org/tools/install). See [development](https://github.com/ScanMountGoat/xenoblade_blender/blob/main/DEVELOPMENT.md) for details.
