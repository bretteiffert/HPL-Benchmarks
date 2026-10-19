use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set"),
    );
    let energy_dir = manifest_dir
        .join("../../dependencies")
        .canonicalize()
        .expect("could not resolve ../../dependencies from the Cargo project directory");
    let energy_library = energy_dir.join("libkernelenergy.so");

    if !energy_library.is_file() {
        panic!("libkernelenergy.so not found at {}", energy_library.display());
    }

    println!("cargo:rustc-link-search=native={}", energy_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", energy_dir.display());
    println!("cargo:rerun-if-changed={}", energy_library.display());
}
