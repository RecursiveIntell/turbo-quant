fn main() {
    cc::Build::new()
        .file("c-kernels/fwht.c")
        .file("c-kernels/qjl.c")
        .file("c-kernels/polar.c")
        .flag_if_supported("-O3")
        .flag_if_supported("-mavx2")
        .flag_if_supported("-mfma")
        .compile("turbo_quant_kernels");

    println!("cargo:rerun-if-changed=c-kernels/turbo_quant.h");
    println!("cargo:rerun-if-changed=c-kernels/fwht.c");
    println!("cargo:rerun-if-changed=c-kernels/bitpack.c");
    println!("cargo:rerun-if-changed=c-kernels/qjl.c");
    println!("cargo:rerun-if-changed=c-kernels/polar.c");
}
