# xenoblade_blender
A Blender addon for importing Xenoblade models and maps from Xenoblade Chronicles 1 DE, Xenoblade Chronicles 2, and Xenoblade Chronicles 3.

## Building
Clone the repository with `git clone https://github.com/ScanMountGoat/xenoblade_blender --recursive`. 

xenoblade_blender uses [xc3_model_py](https://github.com/ScanMountGoat/xc3_model_py) for simplifying the addon code and achieving better performance than pure Python addons. xc3_model_py must be compiled from source after [installing Rust](https://www.rust-lang.org/tools/install). Build the project with `cargo build --release`. This will compile a native Python module for the current Python interpreter. The resulting file in `xc3_model_py/target/release` will need to be copied to the addon folder and renamed to `.pyd` on Windows and `.so` on Linux or MacOS. Make sure to build for the Python version used by Blender. This can be achieved by activating a virtual environment with the appropriate Python version or setting the Python interpeter using the `PYO3_PYTHON` environment variable.