//! What this installer version installs: pinned model revisions, the prebuilt engine asset, cuBLAS.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// GitHub "owner/name" of the repository whose Releases hold the prebuilt engine. Set at build time with the
/// INKLING_REPO environment variable (the release workflow passes GITHUB_REPOSITORY).
pub const REPO: &str = match option_env!("INKLING_REPO") {
    Some(r) => r,
    None => "OWNER/quill-inkling",
};

pub fn repo_known() -> bool {
    !REPO.starts_with("OWNER/")
}

/// Prebuilt engine: ingine-serve, ingine-bench, quill-kld for sm_89 (RTX 40) and sm_120a (RTX 50).
pub const ENGINE_ASSET: &str = "quill-engine-win64-cuda13-sm89-sm120a.zip";
pub const ENGINE_ARCHS: &[(u32, u32)] = &[(8, 9), (12, 0)];

pub fn engine_url() -> String {
    format!("https://github.com/{REPO}/releases/download/v{VERSION}/{ENGINE_ASSET}")
}
pub fn sums_url() -> String {
    format!("https://github.com/{REPO}/releases/download/v{VERSION}/SHA256SUMS.txt")
}
pub fn source_url() -> String {
    format!("https://github.com/{REPO}/archive/refs/tags/v{VERSION}.zip")
}

/// cuBLASLt, the one CUDA library the engine links dynamically (the CUDA runtime is static). Downloaded from
/// NVIDIA's redistributable archive of the toolkit the engine is built with (CUDA 13.3.1 -> cuBLAS 13.6.0.2).
pub const CUBLAS_VERSION: &str = "13.6.0.2";
pub const CUBLAS_URL: &str = "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.6.0.2-archive.zip";
pub const CUBLAS_SHA256: &str = "62e9fa30560c8f0a28e0cdcf9d6fc1fed347bcfab8847239b9ae1fdc1d86408a";
pub const CUBLAS_SIZE: u64 = 393_706_755;
pub const CUBLAS_DLL: &str = "cublasLt64_13.dll";

/// The engine is built with CUDA 13: the driver must support CUDA 13.0 or newer (R580+).
pub const MIN_DRIVER_CUDA: (u32, u32) = (13, 0);

pub struct ModelRepo {
    pub repo: &'static str,
    pub dir: &'static str,
    /// files to fetch: exact paths, or prefixes ending in '/'; empty = the whole repository
    pub include: &'static [&'static str],
}

pub const MODELS: &[ModelRepo] = &[
    ModelRepo { repo: "thinkingmachines/Inkling-2-Mini", dir: "Inkling-2-Mini", include: &[] },
    ModelRepo { repo: "incoai/Inkling-2-Mini-DFlash2", dir: "Inkling-2-Mini-DFlash2", include: &[] },
    ModelRepo {
        repo: "satellitedown/Inkling-2-Mini-abliterated-ingine",
        dir: "ablation-ingine",
        include: &["inkling-2-mini-abliterated.safetensors", "LICENSE", "NOTICE", "README.md", "SHA256SUMS"],
    },
];

/// models/PINS.txt: "<repo> <revision>" per line.
pub const PINS: &str = include_str!("../../models/PINS.txt");

pub fn pinned_revision(repo: &str) -> Option<&'static str> {
    PINS.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        (it.next()? == repo).then(|| it.next()).flatten()
    })
}

/// Approximate download size of the models (for the disk-space check and the UI).
pub const MODELS_BYTES: u64 = 21_200_000_000;

pub const CHAT_HTML: &str = include_str!("../../app/chat.html");

#[cfg(test)]
mod tests {
    #[test]
    fn every_model_is_pinned() {
        for m in super::MODELS {
            let rev = super::pinned_revision(m.repo).unwrap_or_else(|| panic!("{} not in PINS.txt", m.repo));
            assert_eq!(rev.len(), 40, "{rev}");
        }
    }
}
