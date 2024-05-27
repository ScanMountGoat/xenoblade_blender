# xenoblade_blender [![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ScanMountGoat/xenoblade_blender?include_prereleases)](https://github.com/ScanMountGoat/xenoblade_blender/releases/latest)
A Blender addon for importing models, maps, and animations for Xenoblade Chronicles X, Xenoblade Chronicles 1 DE, Xenoblade Chronicles 2, and Xenoblade Chronicles 3. 

Report bugs or request new features in [issues](https://github.com/ScanMountGoat/xenoblade_blender/issues). Download the latest version from [releases](https://github.com/ScanMountGoat/xenoblade_blender/releases). Check the [wiki](https://github.com/ScanMountGoat/xenoblade_blender/wiki) for more usage information.

## Features
* Faster and more accurate than intermediate formats like FBX or glTF
* Import characters, objects, enemies, and weapon models from .wimdo and .camdo files
* Import maps from .wismhd files
* Import character animations from .mot files
* Decoded and embedded RGBA textures for easier use in Blender
* Material parameters, textures, and texture channel assignments based on decompiled in game shader code
* Export .wimdo models using an existing model as a base (WIP)

## Getting Started
* Download the latest version of the addon supported by your Blender version from [releases](https://github.com/ScanMountGoat/xenoblade_blender/releases).
* Install the .zip file in Blender using Edit > Preferences > Addons > Install...
* Enable the addon if it is not already enabled.
* Import any of the supported file types using the new menu options under File > Import.
* Select the appropriate game version before importing for correct material texture assignments.

## Planned Features
* Improved wimdo export
* Exporting animations
* Better support for Xenoblade X models and maps
* Improvements to import times and memory usage

The goal of this addon is to provide import and export operations with good performance and minimal configuration. Quality of life features requiring more complex UI are better handled using traditional pure Python addons.

## Building
Clone the repository with `git clone https://github.com/ScanMountGoat/xenoblade_blender --recursive`. 

xenoblade_blender uses [xc3_model_py](https://github.com/ScanMountGoat/xc3_model_py) for simplifying the addon code and achieving better performance than pure Python addons. 

xc3_model_py must be compiled from source after [installing Rust](https://www.rust-lang.org/tools/install). Build the project from the xc3_model_py directory with `cargo build --release`. This will compile a native Python module for the current Python interpreter. 

The resulting file in `xc3_model_py/target/release` will need to be copied to the addon folder and renamed to `.pyd` on Windows and `.so` on Linux or MacOS. 

Make sure to build for the Python version used by Blender. This can be achieved by activating a virtual environment with the appropriate Python version or setting the Python interpeter using the `PYO3_PYTHON` environment variable.
