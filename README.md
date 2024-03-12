# xenoblade_blender
A Blender addon for importing Xenoblade models and maps from Xenoblade Chronicles 1 DE, Xenoblade Chronicles 2, and Xenoblade Chronicles 3 are supported.

## Building
xenoblade_blender uses [xc3_model_py](https://github.com/ScanMountGoat/xc3_model_py) for simplifying the addon code and achieving better performance than pure Python addons. xc3_model_py must be compiled from source using `cargo build --release` after [installing Rust](https://www.rust-lang.org/tools/install).
