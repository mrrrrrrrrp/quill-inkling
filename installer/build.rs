// Embeds inkling.manifest (asInvoker, common controls v6, long paths) with the MSVC linker.
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest = std::path::Path::new(&dir).join("inkling.manifest");
    println!("cargo:rerun-if-changed=inkling.manifest");
    println!("cargo:rerun-if-env-changed=INKLING_REPO");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:{}", manifest.display());
    }
}
