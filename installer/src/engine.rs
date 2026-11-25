//! The engine binaries in <install>\bin: the prebuilt release zip, or a build from source; plus cuBLASLt.

use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::download::{self, Check, Http};
use crate::progress::Reporter;
use crate::release;
use crate::util::hidden;

pub const EXES: &[&str] = &["ingine-serve.exe", "ingine-bench.exe", "quill-kld.exe"];

#[derive(Clone, Debug)]
pub enum EngineSource {
    /// the release zip from GitHub; `from` = a local copy of that zip or a folder holding the exes
    Prebuilt { from: Option<PathBuf> },
    /// build engine\ for `arch` (CMAKE_CUDA_ARCHITECTURES); `dir` = a checkout of the repository
    Source { dir: Option<PathBuf>, arch: String },
}

impl EngineSource {
    pub fn describe(&self) -> String {
        match self {
            EngineSource::Prebuilt { .. } => "prebuilt sm_89 + sm_120a".into(),
            EngineSource::Source { arch, .. } => format!("built from source for {arch}"),
        }
    }
}

pub fn install(http: &Http, src: &EngineSource, root: &Path, rep: &dyn Reporter) -> Result<()> {
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin)?;
    match src {
        EngineSource::Prebuilt { from: Some(p) } if p.is_dir() => copy_exes(p, &bin, rep),
        EngineSource::Prebuilt { from: Some(zip) } => extract_exes(zip, &bin, rep),
        EngineSource::Prebuilt { from: None } => {
            if !release::repo_known() {
                bail!(
                    "this installer was built without its GitHub repository (INKLING_REPO), so it cannot fetch the \
                     prebuilt engine; use --engine-from <zip or folder> or --from-source"
                );
            }
            let (sums, _) = http.get_text(&release::sums_url()).context("fetching the release checksums")?;
            let sha = sums
                .lines()
                .find_map(|l| {
                    let mut it = l.split_whitespace();
                    let (h, name) = (it.next()?, it.next()?);
                    (name.trim_start_matches('*') == release::ENGINE_ASSET).then(|| h.to_string())
                })
                .with_context(|| format!("{} is not listed in SHA256SUMS.txt", release::ENGINE_ASSET))?;
            let zip = root.join("downloads").join(release::ENGINE_ASSET);
            http.download(&release::engine_url(), &zip, None, &Check::Sha256(sha), "engine", rep)?;
            extract_exes(&zip, &bin, rep)?;
            let _ = std::fs::remove_file(&zip);
            Ok(())
        }
        EngineSource::Source { dir, arch } => build(http, dir.as_deref(), arch, root, rep),
    }
}

fn extract_exes(zip: &Path, bin: &Path, rep: &dyn Reporter) -> Result<()> {
    let got = download::unzip(zip, |name| {
        let file = name.rsplit('/').next()?;
        (file.ends_with(".exe") || file.ends_with(".dll")).then(|| bin.join(file))
    })?;
    for p in &got {
        rep.log(&format!("installed {}", p.display()));
    }
    check_exes(bin)
}

/// From a folder with the exes, or a CMake build folder (apps\serve\..., apps\bench\..., apps\kld\...).
fn copy_exes(from: &Path, bin: &Path, rep: &dyn Reporter) -> Result<()> {
    for exe in EXES {
        let sub = match *exe {
            "ingine-serve.exe" => "serve",
            "ingine-bench.exe" => "bench",
            _ => "kld",
        };
        let cand = [from.join(exe), from.join("bin").join(exe), from.join("apps").join(sub).join(exe)];
        let src = cand.iter().find(|p| p.is_file()).with_context(|| format!("{exe} not found under {}", from.display()))?;
        replace_copy(src, &bin.join(exe))?;
        rep.log(&format!("installed {} (from {})", exe, src.display()));
    }
    Ok(())
}

fn check_exes(bin: &Path) -> Result<()> {
    if !bin.join("ingine-serve.exe").is_file() {
        bail!("ingine-serve.exe is missing from {}", bin.display());
    }
    Ok(())
}

fn replace_copy(src: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        std::fs::remove_file(dest).with_context(|| format!("replacing {} (is Inkling running?)", dest.display()))?;
    }
    std::fs::copy(src, dest).with_context(|| format!("copying {}", src.display()))?;
    Ok(())
}

/// cuBLASLt next to the engine: the toolkit's copy for a source build, NVIDIA's redistributable otherwise.
pub fn install_cublas(http: &Http, root: &Path, from_toolkit: bool, rep: &dyn Reporter) -> Result<()> {
    let bin = root.join("bin");
    let dll = bin.join(release::CUBLAS_DLL);
    let marker = bin.join("cublas.version");
    if from_toolkit {
        let tc = Toolchain::detect();
        let cuda = tc.cuda_path.context("CUDA_PATH is not set")?;
        let src = [cuda.join("bin").join("x64").join(release::CUBLAS_DLL), cuda.join("bin").join(release::CUBLAS_DLL)]
            .into_iter()
            .find(|p| p.is_file())
            .with_context(|| format!("{} not found in {}", release::CUBLAS_DLL, cuda.display()))?;
        replace_copy(&src, &dll)?;
        std::fs::write(&marker, format!("toolkit {}", tc.nvcc_version.unwrap_or_default()))?;
        rep.log(&format!("copied {} from {}", release::CUBLAS_DLL, src.display()));
        return Ok(());
    }
    if dll.is_file() && std::fs::read_to_string(&marker).map(|v| v.trim() == release::CUBLAS_VERSION).unwrap_or(false) {
        rep.log(&format!("cuBLAS {} already installed", release::CUBLAS_VERSION));
        return Ok(());
    }
    let zip = root.join("downloads").join(release::CUBLAS_URL.rsplit('/').next().unwrap());
    http.download(
        release::CUBLAS_URL,
        &zip,
        Some(release::CUBLAS_SIZE),
        &Check::Sha256(release::CUBLAS_SHA256.into()),
        &format!("cuBLAS {} (NVIDIA)", release::CUBLAS_VERSION),
        rep,
    )?;
    download::unzip(&zip, |name| name.ends_with(&format!("/{}", release::CUBLAS_DLL)).then(|| dll.clone()))?;
    std::fs::write(&marker, release::CUBLAS_VERSION)?;
    let _ = std::fs::remove_file(&zip);
    rep.log(&format!("installed {}", dll.display()));
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct Toolchain {
    pub vs: Option<PathBuf>,
    pub cuda_path: Option<PathBuf>,
    pub nvcc_version: Option<String>,
}

impl Toolchain {
    pub fn detect() -> Toolchain {
        let vswhere = PathBuf::from(std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".into()))
            .join(r"Microsoft Visual Studio\Installer\vswhere.exe");
        let vs = hidden(Command::new(&vswhere).args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]))
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
        let cuda_path = std::env::var("CUDA_PATH").ok().map(PathBuf::from).filter(|p| p.join("bin").join("nvcc.exe").is_file());
        let nvcc_version = cuda_path.as_ref().and_then(|c| {
            let o = hidden(Command::new(c.join("bin").join("nvcc.exe")).arg("--version")).output().ok()?;
            let t = String::from_utf8_lossy(&o.stdout).into_owned();
            let i = t.find("release ")?;
            Some(t[i + 8..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect())
        });
        Toolchain { vs, cuda_path, nvcc_version }
    }

    /// What is missing for a source build, with the command that installs it.
    pub fn missing(&self) -> Vec<String> {
        let mut m = Vec::new();
        if self.vs.is_none() {
            m.push(
                "Visual Studio 2022 Build Tools with C++: winget install --id Microsoft.VisualStudio.2022.BuildTools \
                 --override \"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait\""
                    .into(),
            );
        }
        match &self.nvcc_version {
            None => m.push("CUDA Toolkit 13.x: winget install --id Nvidia.CUDA (then sign out and in, so CUDA_PATH is set)".into()),
            Some(v) if !v.starts_with("13.") => m.push(format!("CUDA Toolkit 13.x (found {v}): winget install --id Nvidia.CUDA")),
            _ => {}
        }
        m
    }
}

/// A checkout of this repository near the running exe or the current folder (engine\CMakeLists.txt).
pub fn find_checkout() -> Option<PathBuf> {
    let mut starts = Vec::new();
    if let Ok(e) = std::env::current_exe() {
        starts.push(e);
    }
    if let Ok(c) = std::env::current_dir() {
        starts.push(c.join("x"));
    }
    for s in starts {
        for a in s.ancestors().skip(1).take(6) {
            if a.join("engine").join("CMakeLists.txt").is_file() && a.join("installer").join("Cargo.toml").is_file() {
                return Some(a.to_path_buf());
            }
        }
    }
    None
}

fn build(http: &Http, dir: Option<&Path>, arch: &str, root: &Path, rep: &dyn Reporter) -> Result<()> {
    let tc = Toolchain::detect();
    let missing = tc.missing();
    if !missing.is_empty() {
        bail!("a source build needs:\n  - {}", missing.join("\n  - "));
    }
    let src = match dir.map(Path::to_path_buf).or_else(find_checkout) {
        Some(d) => d,
        None => fetch_source(http, root, rep)?,
    };
    let engine = src.join("engine");
    if !engine.join("CMakeLists.txt").is_file() {
        bail!("{} has no engine\\CMakeLists.txt", src.display());
    }
    let bdir = format!("build-{}", arch.replace(';', "-"));
    rep.log(&format!("building {} for {arch} with CUDA {} (a few minutes)", engine.display(), tc.nvcc_version.unwrap_or_default()));
    let script = engine.join("build.cmd");
    let mut child = hidden(
        Command::new("cmd")
            .raw_arg(format!("/d /s /c \"\"{}\" \"\" {bdir} 2>&1\"", script.display()))
            .current_dir(&engine)
            .env("QUILL_CUDA_ARCH", arch)
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .spawn()
    .context("starting build.cmd")?;
    let mut tail: Vec<String> = Vec::new();
    for line in BufReader::new(child.stdout.take().unwrap()).lines().map_while(|l| l.ok()) {
        if rep.cancelled() {
            let _ = child.kill();
            return Err(crate::progress::Cancelled.into());
        }
        rep.log(&line);
        tail.push(line);
        if tail.len() > 40 {
            tail.remove(0);
        }
    }
    let st = child.wait()?;
    if !st.success() {
        bail!("the engine build failed (exit {:?}); last lines:\n{}", st.code(), tail.join("\n"));
    }
    copy_exes(&engine.join(&bdir), &root.join("bin"), rep)
}

fn fetch_source(http: &Http, root: &Path, rep: &dyn Reporter) -> Result<PathBuf> {
    if !release::repo_known() {
        bail!("no source checkout found; pass --source <folder of the repository>");
    }
    let zip = root.join("downloads").join(format!("source-v{}.zip", release::VERSION));
    http.download(&release::source_url(), &zip, None, &Check::None, "engine source", rep)?;
    let dest = root.join("src");
    download::unzip(&zip, |name| {
        // strip GitHub's top folder (<repo>-<version>/)
        let rest = name.split_once('/')?.1;
        (!rest.is_empty()).then(|| dest.join(rest.replace('/', "\\")))
    })?;
    let _ = std::fs::remove_file(&zip);
    Ok(dest)
}
