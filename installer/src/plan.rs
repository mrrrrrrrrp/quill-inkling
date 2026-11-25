//! From detected hardware to a verdict and the settings to install with. AGENTS.md explains the numbers.

use serde::Serialize;

use crate::config::{Server, Vision};
use crate::hw::{Gpu, Hardware};
use crate::release;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Support {
    /// the reference setup: RTX 40 with 12 GB + 32 GB RAM (100+ tok/s, 600k context)
    Confirmed,
    /// should work with the prebuilt engine, not confirmed yet
    Expected,
    /// may work; untested, slower, or needs a source build
    Experimental,
    /// will not work without engine changes (AGENTS.md)
    Unsupported,
}

impl Support {
    pub fn label(self) -> &'static str {
        match self {
            Support::Confirmed => "Confirmed setup",
            Support::Expected => "Should work (not confirmed yet)",
            Support::Experimental => "Experimental",
            Support::Unsupported => "Not supported as is",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Assessment {
    pub support: Support,
    pub summary: String,
    pub notes: Vec<String>,
    /// the GPU the engine will use
    pub gpu: Option<Gpu>,
    /// CUDA architecture to build for (CMAKE_CUDA_ARCHITECTURES), e.g. "89", "120a"
    pub cuda_arch: Option<String>,
    /// the prebuilt engine covers this GPU
    pub prebuilt: bool,
    pub settings: Server,
    /// largest context the GPU is estimated to hold (tokens)
    pub max_ctx_fit: u32,
}

/// Context presets offered by the installer (tokens). 600064 is the confirmed default.
pub const CTX_PRESETS: &[u32] = &[131_072, 262_144, 600_064, 1_048_576];
pub const CTX_DEFAULT: u32 = 600_064;

// VRAM model of the engine (measured on the reference machine, see AGENTS.md "Memory"):
const OTHER_APPS_GIB: f64 = 1.0; // desktop, browser: counted against the cap (all processes)
const ENGINE_FIXED_GIB: f64 = 2.1; // dense weights, drafter, CUDA context, buffers
const REGION_RESERVED_GIB: f64 = 0.95; // prefill scratch (806 MiB) + 24 staging slots, inside the shared region
const KV_BYTES_PER_TOKEN: f64 = 9216.0; // q4 full-attention KV at 3 loop passes

pub fn vram_cap_gib(gpu: &Gpu) -> f64 {
    // card size minus ~0.7 GB, one decimal: 12282 MiB -> 11.3
    ((gpu.vram_gib() - 0.7) * 10.0).round() / 10.0
}

pub fn max_ctx_fit(cap_gib: f64) -> u32 {
    let kv_gib = cap_gib - OTHER_APPS_GIB - ENGINE_FIXED_GIB - REGION_RESERVED_GIB;
    if kv_gib <= 0.0 {
        return 0;
    }
    let t = (kv_gib * 1024.0 * 1024.0 * 1024.0 / KV_BYTES_PER_TOKEN) as u32;
    t / 64 * 64
}

pub fn default_ctx(fit: u32) -> u32 {
    CTX_PRESETS.iter().copied().filter(|&c| c <= fit.min(CTX_DEFAULT)).max().unwrap_or(65_536.min(fit / 64 * 64))
}

pub fn cuda_arch(cc: (u32, u32)) -> String {
    match cc {
        (8, 9) => "89".into(),
        // Hopper and Blackwell builds use the arch-specific targets the upstream kernels were written for
        (a, b) if a >= 9 => format!("{a}{b}a"),
        (a, b) => format!("{a}{b}"),
    }
}

pub fn assess(hw: &Hardware) -> Assessment {
    let mut notes = Vec::new();
    let mut support = Support::Confirmed;
    let mut lower = |s: Support| {
        if (s as u8) > (support as u8) {
            support = s;
        }
    };

    // the GPU: the largest one that is new enough, else the largest
    let gpu = hw
        .gpus
        .iter()
        .filter(|g| g.cc >= (8, 9))
        .max_by_key(|g| g.vram_mib)
        .or_else(|| hw.gpus.iter().max_by_key(|g| g.vram_mib))
        .cloned();

    let ram = hw.ram_gib();
    let threads = hw.physical_cores.saturating_sub(1).max(1);
    let mut settings = Server::default();
    settings.cpu_threads = threads;
    settings.vision_threads = hw.physical_cores.clamp(2, 16);

    let Some(g) = gpu.clone() else {
        notes.push(format!(
            "No NVIDIA GPU found{}. quill needs an NVIDIA RTX 40 or 50 series card.",
            hw.gpu_error.as_ref().map(|e| format!(" ({e})")).unwrap_or_default()
        ));
        return Assessment {
            support: Support::Unsupported,
            summary: "No NVIDIA GPU detected.".into(),
            notes,
            gpu: None,
            cuda_arch: None,
            prebuilt: false,
            settings,
            max_ctx_fit: 0,
        };
    };

    let cap = vram_cap_gib(&g);
    let fit = max_ctx_fit(cap);
    settings.vram_cap_gib = cap;
    settings.max_ctx = default_ctx(fit);
    settings.vision = if cap >= 15.0 { Vision::Gpu } else { Vision::Cpu };
    let arch = cuda_arch(g.cc);
    let prebuilt = release::ENGINE_ARCHS.contains(&g.cc);
    let vram = g.vram_gib();

    // GPU generation
    match g.cc {
        (8, 9) => {}
        (12, 0) => {
            lower(Support::Expected);
            notes.push(
                "RTX 50 (Blackwell): the prebuilt engine includes sm_120a code, but the kernels were tuned on an RTX 40 \
                 and this has not been confirmed on an RTX 50 yet."
                    .into(),
            );
        }
        cc if cc >= (8, 9) => {
            lower(Support::Experimental);
            notes.push(format!(
                "{} (compute {}.{}) is not covered by the prebuilt engine: it has to be built from source for {arch}, untested.",
                g.name, cc.0, cc.1
            ));
        }
        cc => {
            lower(Support::Unsupported);
            notes.push(format!(
                "{} is compute {}.{}: the kernels need compute 8.9+ (RTX 40 or newer) for FP8 conversions. \
                 Porting notes are in AGENTS.md.",
                g.name, cc.0, cc.1
            ));
        }
    }

    // driver
    match hw.driver_cuda {
        Some(v) if v < release::MIN_DRIVER_CUDA => {
            lower(Support::Unsupported);
            notes.push(format!(
                "The NVIDIA driver supports CUDA {}.{}; the engine needs CUDA 13 (driver R580 or newer). Update the driver.",
                v.0, v.1
            ));
        }
        None => notes.push("Could not read the driver's CUDA version; it needs to be 13.0 or newer (R580+).".into()),
        _ => {}
    }

    // VRAM
    if vram < 7.5 {
        lower(Support::Unsupported);
        notes.push(format!("{vram:.0} GB of VRAM is too little for the dense weights, the drafter and a useful KV cache."));
    } else if vram < 11.5 {
        lower(Support::Experimental);
        notes.push(format!(
            "{vram:.0} GB of VRAM: below the 12 GB reference. Fewer experts fit on the GPU, so decoding is slower, \
             and the context is limited to {}k.",
            settings.max_ctx / 1000
        ));
    } else if vram >= 13.0 {
        if g.cc == (8, 9) {
            lower(Support::Expected);
        }
        notes.push(format!(
            "{vram:.0} GB of VRAM: more than the 12 GB reference, so more experts stay on the GPU; expect it to be faster. \
             Not confirmed yet.{}",
            if vram >= 19.0 {
                " Every expert may fit on the GPU, a path the engine has not been tested on (AGENTS.md)."
            } else {
                ""
            }
        ));
    }

    // system RAM: every expert is held in pinned host memory (~16 GiB with the embedding and drafter tables)
    if ram < 24.0 {
        lower(Support::Unsupported);
        notes.push(format!(
            "{ram:.0} GB of RAM: the engine keeps all experts (~16 GB) in pinned system memory and needs 32 GB. \
             AGENTS.md describes the changes for less RAM."
        ));
    } else if ram < 30.0 {
        lower(Support::Experimental);
        notes.push(format!(
            "{ram:.0} GB of RAM: below the 32 GB reference; ~16 GB is pinned for the experts, so loading may fail or \
             the system may page. Close other programs."
        ));
    }

    // CPU
    if !hw.avx2 {
        lower(Support::Unsupported);
        notes.push("The CPU lacks AVX2, which the CPU expert tier needs.".into());
    }
    if hw.physical_cores < 6 {
        lower(Support::Experimental);
        notes.push(format!(
            "{} CPU cores: the experts that do not fit on the GPU run on the CPU; fewer than 6 cores will slow decoding.",
            hw.physical_cores
        ));
    }

    if hw.gpus.len() > 1 {
        notes.push(format!("{} GPUs found; the engine will use GPU {} ({}).", hw.gpus.len(), g.index, g.name));
    }
    if fit >= 1_048_576 {
        notes.push("This GPU can hold the model's full 1M-token context (choose it under Context).".into());
    }

    let summary = match support {
        Support::Confirmed => {
            "This is the confirmed reference setup: RTX 40 with 12 GB + 32 GB RAM: 100+ tok/s with a 600k context.".into()
        }
        Support::Expected => format!("{} should work; the settings below are scaled from the reference setup.", g.name),
        Support::Experimental => {
            "This may work, but it is untested. Point a coding agent at AGENTS.md to tune it for this machine.".into()
        }
        Support::Unsupported => {
            "This machine cannot run the engine as is. AGENTS.md explains what would have to change.".into()
        }
    };

    Assessment { support, summary, notes, gpu: Some(g), cuda_arch: Some(arch), prebuilt, settings, max_ctx_fit: fit }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(name: &str, mib: u64, cc: (u32, u32), ram_gib: f64, cores: u32) -> Hardware {
        Hardware {
            gpus: vec![Gpu { index: 0, name: name.into(), vram_mib: mib, used_mib: 1000, cc, driver: "591.86".into() }],
            driver_cuda: Some((13, 1)),
            ram_bytes: (ram_gib * 1024.0 * 1024.0 * 1024.0) as u64,
            cpu_name: "test".into(),
            physical_cores: cores,
            logical_cores: cores * 2,
            avx2: true,
            gpu_error: None,
        }
    }

    #[test]
    fn reference_machine_is_confirmed_with_reference_settings() {
        let a = assess(&hw("NVIDIA GeForce RTX 4070 SUPER", 12282, (8, 9), 31.9, 8));
        assert_eq!(a.support, Support::Confirmed, "{:?}", a.notes);
        assert!(a.prebuilt);
        assert_eq!(a.settings.vram_cap_gib, 11.3);
        assert_eq!(a.settings.cpu_threads, 7);
        assert_eq!(a.settings.max_ctx, 600_064);
        assert_eq!(a.settings.vision, Vision::Cpu);
        assert_eq!(a.cuda_arch.as_deref(), Some("89"));
        // the measured ceiling on this machine is ~850k
        assert!((800_000..900_000).contains(&a.max_ctx_fit), "{}", a.max_ctx_fit);
    }

    #[test]
    fn rtx50_16gb_is_expected_and_prebuilt() {
        let a = assess(&hw("NVIDIA GeForce RTX 5080", 16303, (12, 0), 63.8, 8));
        assert_eq!(a.support, Support::Expected);
        assert!(a.prebuilt);
        assert_eq!(a.cuda_arch.as_deref(), Some("120a"));
        assert_eq!(a.settings.max_ctx, 600_064);
        assert!(a.max_ctx_fit >= 1_048_576);
    }

    #[test]
    fn rtx4090_is_expected() {
        let a = assess(&hw("NVIDIA GeForce RTX 4090", 24564, (8, 9), 63.8, 16));
        assert_eq!(a.support, Support::Expected);
        assert_eq!(a.settings.vision, Vision::Gpu);
        assert_eq!(a.settings.cpu_threads, 15);
    }

    #[test]
    fn rtx30_is_unsupported() {
        let a = assess(&hw("NVIDIA GeForce RTX 3090", 24576, (8, 6), 63.8, 8));
        assert_eq!(a.support, Support::Unsupported);
        assert!(!a.prebuilt);
    }

    #[test]
    fn low_ram_is_unsupported_and_8gb_gpu_experimental() {
        assert_eq!(assess(&hw("RTX 4070", 12282, (8, 9), 15.9, 8)).support, Support::Unsupported);
        let a = assess(&hw("RTX 4060", 8188, (8, 9), 31.9, 8));
        assert_eq!(a.support, Support::Experimental);
        assert!(a.settings.max_ctx < 600_064 && a.settings.max_ctx > 0);
    }

    #[test]
    fn hopper_needs_a_source_build() {
        let a = assess(&hw("H100", 81559, (9, 0), 128.0, 32));
        assert_eq!(a.support, Support::Experimental);
        assert!(!a.prebuilt);
        assert_eq!(a.cuda_arch.as_deref(), Some("90a"));
    }

    #[test]
    fn no_gpu() {
        let mut h = hw("x", 0, (8, 9), 32.0, 8);
        h.gpus.clear();
        assert_eq!(assess(&h).support, Support::Unsupported);
    }
}
