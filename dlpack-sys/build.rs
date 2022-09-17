use std::{env, error::Error, path::PathBuf, result::Result};

fn main() -> Result<(), Box<dyn Error>> {
    let bindings = bindgen::Builder::default()
        .header("csrc/dlpack/include/dlpack/dlpack.h")
        .blocklist_type("max_align_t")
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .derive_default(true)
        .generate()
        .expect("Cannot generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Cannot write bindings");
    Ok(())
}
