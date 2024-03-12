# xenoblade_blender
A Blender addon for importing Xenoblade models and maps from Xenoblade Chronicles 1 DE, Xenoblade Chronicles 2, and Xenoblade Chronicles 3 are supported.

## Building
Clone the repository with `git clone https://github.com/ScanMountGoat/xenoblade_blender --recursive`. 

xenoblade_blender uses [xc3_model_py](https://github.com/ScanMountGoat/xc3_model_py) for simplifying the addon code and achieving better performance than pure Python addons. xc3_model_py must be compiled from source after [installing Rust](https://www.rust-lang.org/tools/install). Build the project with `cargo build --release`. This will compile a native python module for the current Python interpreter. For use with Blender, make sure to build for the Python version used by Blender. The easiest way to do this is to use the Python interpreter bundled with Blender. See the [PyO3 guide](https://pyo3.rs/main/building_and_distribution) for details. Some example commands are listed below for different operating systems. 

**Blender 4.0 on Windows**  
```
set PYO3_PYTHON = "C:\Program Files\Blender Foundation\Blender 4.0\4.0\python\bin\python.exe"
cargo build --release
```

**Blender 4.0 on MacOS**  
```
PYO3_PYTHON="/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.10" cargo build --release
```
