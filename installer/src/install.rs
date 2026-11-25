//! The install steps, shared by the window and the command line.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::download::Http;
use crate::engine::{self, EngineSource};
use crate::hf;
use crate::launch;
use crate::progress::Reporter;
use crate::release;
use crate::system;
use crate::util::{fmt_bytes, GIB};

#[derive(Clone, Debug)]
pub struct Options {
    pub dir: PathBuf,
    pub config: Config,
    pub engine: EngineSource,
    /// an existing models folder to link or copy from instead of downloading
    pub models_from: Option<PathBuf>,
    pub desktop_shortcut: bool,
    pub start_menu: bool,
    /// skip the model download (CI and tests)
    pub skip_models: bool,
}

pub fn default_dir() -> PathBuf {
    system::installed_dir().unwrap_or_else(|| {
        PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_else(|_| r"C:\".into())).join("Inkling")
    })
}

pub fn run(o: &Options, rep: &dyn Reporter) -> Result<()> {
    let http = Http::new();
    let dir = &o.dir;
    for d in ["bin", "app", "models"] {
        std::fs::create_dir_all(dir.join(d)).with_context(|| format!("creating {}", dir.join(d).display()))?;
    }
    if Config::exists(dir) {
        if let Ok(c) = Config::load(dir) {
            if launch::healthy(&c) {
                bail!("Inkling is running from this folder: close its server window first");
            }
        }
    }

    rep.step("Checking disk space");
    let models_dir = dir.join("models");
    let (models_total, mut models_missing) = if o.skip_models { (0, 0) } else { hf::plan_bytes(&http, &models_dir)? };
    if o.models_from.is_some() {
        models_missing = 0; // linked (same drive) or copied; the copy case is checked by the OS
    }
    let need = models_missing + (1.5 * GIB) as u64;
    if let Some(free) = crate::hw::free_disk_bytes(dir) {
        rep.log(&format!("{} free, about {} needed", fmt_bytes(free), fmt_bytes(need)));
        if free < need {
            bail!("not enough disk space in {}: {} free, {} needed", dir.display(), fmt_bytes(free), fmt_bytes(need));
        }
    }

    rep.step(&format!("Engine ({})", o.engine.describe()));
    engine::install(&http, &o.engine, dir, rep)?;

    rep.step("cuBLAS");
    engine::install_cublas(&http, dir, matches!(o.engine, EngineSource::Source { .. }), rep)?;

    if !o.skip_models {
        for m in release::MODELS {
            rep.step(&format!("Model files: {}", m.repo));
            hf::sync_repo(&http, m, &models_dir, o.models_from.as_deref(), rep)?;
        }
    }
    if rep.cancelled() {
        return Err(crate::progress::Cancelled.into());
    }

    rep.step("Settings and chat page");
    std::fs::write(dir.join("app").join("chat.html"), release::CHAT_HTML)?;
    let _ = std::fs::write(models_dir.join("PINS.txt"), release::PINS);
    let mut cfg = o.config.clone();
    cfg.install.version = release::VERSION.into();
    cfg.install.engine = o.engine.describe();
    cfg.save(dir)?;
    rep.log(&format!("wrote {}", Config::path(dir).display()));

    rep.step("Launcher and shortcuts");
    install_self(dir)?;
    if let Err(e) = system::create_shortcuts(dir, o.desktop_shortcut, o.start_menu) {
        rep.log(&format!("could not create the shortcuts: {e:#}"));
    }
    if let Err(e) = system::register_uninstall(dir, models_total + (1u64 << 30)) {
        rep.log(&format!("could not register in Apps & features: {e:#}"));
    }
    let _ = std::fs::remove_dir(dir.join("downloads"));
    rep.log(&format!("done: start Inkling from the Start menu, or run {}", dir.join("Inkling.exe").display()));
    Ok(())
}

/// Copies the running exe to <dir>\Inkling.exe (the launcher and settings program) and writes inkling-cli.exe,
/// the same program as a console program for terminals and scripts.
fn install_self(dir: &Path) -> Result<()> {
    let me = std::env::current_exe()?;
    let dest = dir.join("Inkling.exe");
    let same = match (me.canonicalize(), dest.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same {
        if dest.exists() {
            std::fs::remove_file(&dest).context("replacing Inkling.exe: close the Inkling server window first")?;
        }
        std::fs::copy(&me, &dest).context("copying the launcher")?;
    }
    let cli = dir.join(CLI_EXE);
    let me_cli = me.canonicalize().ok() == cli.canonicalize().ok() && cli.exists();
    if !me_cli {
        system::write_console_copy(&dest, &cli)?;
    }
    Ok(())
}

pub const CLI_EXE: &str = "inkling-cli.exe";

/// Removes what the installer created; other files in the folder are left alone.
pub fn uninstall(dir: &Path, keep_models: bool, rep: &dyn Reporter) -> Result<()> {
    if let Ok(c) = Config::load(dir) {
        if launch::healthy(&c) {
            bail!("Inkling is running: close its server window first");
        }
    }
    rep.step("Removing shortcuts and the Apps & features entry");
    if let Err(e) = system::remove_shortcuts(dir) {
        rep.log(&format!("shortcuts: {e:#}"));
    }
    system::unregister_uninstall();

    rep.step("Removing files");
    let mut dirs: Vec<PathBuf> = ["bin", "app", "downloads", "src"].iter().map(|d| dir.join(d)).collect();
    if !keep_models {
        dirs.extend(release::MODELS.iter().map(|m| dir.join("models").join(m.dir)));
    }
    for d in dirs.iter().filter(|d| d.exists()) {
        rep.log(&format!("deleting {}", d.display()));
        std::fs::remove_dir_all(d).with_context(|| format!("deleting {}", d.display()))?;
    }
    if !keep_models {
        let _ = std::fs::remove_file(dir.join("models").join("PINS.txt"));
        let _ = std::fs::remove_dir(dir.join("models"));
    }
    let _ = std::fs::remove_file(Config::path(dir));
    let me = std::env::current_exe().ok().and_then(|p| p.canonicalize().ok());
    let mut later = None;
    for exe in [dir.join("Inkling.exe"), dir.join(CLI_EXE)] {
        if !exe.exists() {
            continue;
        }
        if me.is_some() && me == exe.canonicalize().ok() {
            later = Some(exe); // the running program: deleted after it exits
        } else {
            let _ = std::fs::remove_file(&exe);
        }
    }
    match later {
        Some(exe) => system::delete_self_later(&exe, dir),
        None => {
            let _ = std::fs::remove_dir(dir);
        }
    }
    rep.log(if keep_models { "removed (the models were kept)" } else { "removed" });
    Ok(())
}
