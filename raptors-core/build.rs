fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("Cargo must set OUT_DIR for build scripts");
    let header_path = std::path::PathBuf::from(out_dir).join("raptors_core.h");
    // Generate C bindings using cbindgen
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_header("/* Generated C header for raptors-core */")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(header_path);
    
    // Tell cargo to re-run this build script if cbindgen.toml changes
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/ffi");
}
