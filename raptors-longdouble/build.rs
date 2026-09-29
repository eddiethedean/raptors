use std::env;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(raptors_native_longdouble)");
    println!("cargo:rerun-if-changed=src/longdouble.c");
    println!("cargo:rerun-if-env-changed=RAPTORS_SKIP_NATIVE_LONGDOUBLE");

    let target = env::var("TARGET").expect("Cargo sets TARGET");
    if env::var_os("RAPTORS_SKIP_NATIVE_LONGDOUBLE").is_some() {
        return;
    }
    let supported = target.contains("linux")
        && (target.starts_with("x86_64-") || target.starts_with("aarch64-"))
        || target == "x86_64-apple-darwin";
    if !supported {
        return;
    }

    cc::Build::new()
        .file("src/longdouble.c")
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("raptors_longdouble");
    println!("cargo:rustc-cfg=raptors_native_longdouble");
}
