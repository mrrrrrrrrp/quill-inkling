//! Hardware detection: NVIDIA GPUs (nvidia-smi), system RAM, CPU cores and AVX2, free disk space.

use serde::Serialize;
use std::path::Path;
use std::process::Command;

use crate::util::{hidden, GIB};

#[derive(Clone, Debug, Serialize)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub vram_mib: u64,
    pub used_mib: u64,
    /// compute capability (major, minor): 8.9 = RTX 40, 12.0 = RTX 50
    pub cc: (u32, u32),
    pub driver: String,
}

impl Gpu {
    pub fn vram_gib(&self) -> f64 {
        self.vram_mib as f64 / 1024.0
    }
    pub fn sm(&self) -> String {
        format!("sm_{}{}", self.cc.0, self.cc.1)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Hardware {
    pub gpus: Vec<Gpu>,
    /// highest CUDA version the driver supports (nvidia-smi header), e.g. (13, 1)
    pub driver_cuda: Option<(u32, u32)>,
    pub ram_bytes: u64,
    pub cpu_name: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub avx2: bool,
    /// why GPU detection failed, if it did
    pub gpu_error: Option<String>,
}

impl Hardware {
    pub fn ram_gib(&self) -> f64 {
        self.ram_bytes as f64 / GIB
    }
}

pub fn detect() -> Hardware {
    let (gpus, gpu_error) = match query_gpus() {
        Ok(g) if g.is_empty() => (g, Some("nvidia-smi lists no GPU".to_string())),
        Ok(g) => (g, None),
        Err(e) => (Vec::new(), Some(e)),
    };
    let (physical_cores, logical_cores) = cpu_cores();
    Hardware {
        gpus,
        driver_cuda: driver_cuda_version(),
        ram_bytes: total_ram(),
        cpu_name: cpu_name(),
        physical_cores,
        logical_cores,
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        gpu_error,
    }
}

fn nvidia_smi() -> String {
    let sys = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
    let p = Path::new(&sys).join("System32").join("nvidia-smi.exe");
    if p.exists() { p.to_string_lossy().into_owned() } else { "nvidia-smi".into() }
}

fn query_gpus() -> Result<Vec<Gpu>, String> {
    let out = hidden(Command::new(nvidia_smi()).args([
        "--query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]))
    .output()
    .map_err(|e| format!("nvidia-smi not found ({e}): no NVIDIA driver installed?"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(parse_gpus(&String::from_utf8_lossy(&out.stdout)))
}

pub fn parse_gpus(csv: &str) -> Vec<Gpu> {
    csv.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            if f.len() < 6 {
                return None;
            }
            let (ma, mi) = f[4].split_once('.')?;
            Some(Gpu {
                index: f[0].parse().ok()?,
                name: f[1].to_string(),
                vram_mib: f[2].parse().ok()?,
                used_mib: f[3].parse().unwrap_or(0),
                cc: (ma.parse().ok()?, mi.parse().ok()?),
                driver: f[5].to_string(),
            })
        })
        .collect()
}

fn driver_cuda_version() -> Option<(u32, u32)> {
    let out = hidden(&mut Command::new(nvidia_smi())).output().ok()?;
    parse_driver_cuda(&String::from_utf8_lossy(&out.stdout))
}

pub fn parse_driver_cuda(text: &str) -> Option<(u32, u32)> {
    let i = text.find("CUDA Version:")?;
    let v = text[i + 13..].trim_start();
    let v: String = v.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let (a, b) = v.split_once('.')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn total_ram() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { m.ullTotalPhys } else { 0 }
}

fn cpu_cores() -> (u32, u32) {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let logical = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1);
    let mut len = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len) };
    if len == 0 {
        return (logical, logical);
    }
    let mut buf = vec![0u8; len as usize];
    let ok = unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut len,
        )
    };
    if ok == 0 {
        return (logical, logical);
    }
    let (mut off, mut cores) = (0usize, 0u32);
    while off + 8 <= len as usize {
        // each record starts with { Relationship: u32, Size: u32 }
        let size = u32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap()) as usize;
        if size == 0 {
            break;
        }
        cores += 1;
        off += size;
    }
    (cores.max(1), logical)
}

fn cpu_name() -> String {
    use winreg::{enums::HKEY_LOCAL_MACHINE, RegKey};
    RegKey::predef(HKEY_LOCAL_MACHINE)
        .open_subkey(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        .and_then(|k| k.get_value::<String, _>("ProcessorNameString"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown CPU".into())
}

/// Free bytes on the volume that holds `path` (the nearest existing ancestor).
pub fn free_disk_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let mut p = Some(path);
    while let Some(q) = p {
        if q.exists() {
            let w: Vec<u16> = q.as_os_str().encode_wide().chain(Some(0)).collect();
            let mut free = 0u64;
            let ok = unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) };
            return if ok != 0 { Some(free) } else { None };
        }
        p = q.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv() {
        let g = parse_gpus("0, NVIDIA GeForce RTX 4070 SUPER, 12282, 2039, 8.9, 591.86\n");
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].cc, (8, 9));
        assert_eq!(g[0].vram_mib, 12282);
        assert_eq!(g[0].sm(), "sm_89");
    }

    #[test]
    fn parses_driver_cuda() {
        let t = "| NVIDIA-SMI 591.86   Driver Version: 591.86   CUDA Version: 13.1     |";
        assert_eq!(parse_driver_cuda(t), Some((13, 1)));
    }
}
