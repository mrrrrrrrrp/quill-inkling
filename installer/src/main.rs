// A windowed program: the installer is a window, the launcher opens its own console for the server log, and the
// command-line subcommands write to the calling terminal (or to a pipe when the output is captured).
#![windows_subsystem = "windows"]

mod config;
mod download;
mod engine;
mod gui;
mod hf;
mod hw;
mod install;
mod launch;
mod plan;
mod progress;
mod release;
mod system;
mod util;

use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use config::{Config, Vision};
use engine::EngineSource;

#[derive(Parser)]
#[command(
    name = "Inkling",
    version,
    about = "Installer and launcher for Inkling 2 Mini on the quill engine (Windows + NVIDIA RTX 40/50).",
    long_about = "Installer and launcher for Inkling 2 Mini on the quill engine (Windows + NVIDIA RTX 40/50).\n\n\
                  Without a subcommand: starts the chat when inkling.toml is next to this program, else opens the installer."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the server in a console window and open the chat page
    Launch {
        /// install folder (default: this program's folder)
        #[arg(long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        no_browser: bool,
    },
    /// Open the installer / settings window
    Setup {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Print the detected hardware, the verdict and the settings an install would use
    Detect {
        /// machine-readable output
        #[arg(long)]
        json: bool,
    },
    /// Install or update without the window
    Install(InstallArgs),
    /// Print the server command line that the launcher runs (from inkling.toml)
    Command {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Remove Inkling (opens the window unless --yes)
    Uninstall {
        #[arg(long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        yes: bool,
        /// keep the downloaded models (~20 GB)
        #[arg(long)]
        keep_models: bool,
    },
}

#[derive(Args)]
struct InstallArgs {
    /// install folder (default: the existing install, else %LOCALAPPDATA%\Inkling)
    #[arg(long)]
    dir: Option<PathBuf>,
    /// context window in tokens (default: 600064 when the GPU holds it)
    #[arg(long)]
    max_ctx: Option<u32>,
    /// GPU memory cap in GiB, all processes (default: card size - 0.7)
    #[arg(long)]
    vram_cap_gib: Option<f64>,
    /// CPU expert threads (default: physical cores - 1)
    #[arg(long)]
    cpu_threads: Option<u32>,
    /// images: cpu, gpu or off
    #[arg(long)]
    vision: Option<String>,
    #[arg(long)]
    port: Option<u16>,
    /// server environment variable KEY=VALUE (repeatable), e.g. QUILL_ASSIST=0.2
    #[arg(long = "env", value_name = "KEY=VALUE")]
    env: Vec<String>,
    /// build the engine from source (Visual Studio 2022 + CUDA 13 needed)
    #[arg(long)]
    from_source: bool,
    /// repository checkout to build from (default: found next to this program, else downloaded)
    #[arg(long)]
    source: Option<PathBuf>,
    /// CUDA architectures for --from-source, e.g. 89, 120a, "89;120a" (default: the GPU's)
    #[arg(long)]
    arch: Option<String>,
    /// use this engine zip or folder instead of downloading the release
    #[arg(long)]
    engine_from: Option<PathBuf>,
    /// link or copy the model files from this folder instead of downloading them
    #[arg(long)]
    models_from: Option<PathBuf>,
    #[arg(long)]
    no_desktop_shortcut: bool,
    #[arg(long)]
    no_start_menu: bool,
    /// do not download the models (tests)
    #[arg(long, hide = true)]
    skip_models: bool,
    /// install even when the hardware check says the machine is not supported
    #[arg(long)]
    force: bool,
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(Path::to_path_buf)
}

fn installed_here() -> Option<PathBuf> {
    exe_dir().filter(|d| Config::exists(d))
}

fn main() -> ExitCode {
    system::note_start_console();
    let args: Vec<String> = std::env::args().collect();
    let wants_console = args.len() > 1 && !matches!(args[1].as_str(), "launch" | "setup");
    if wants_console {
        system::attach_parent_console();
    }
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            system::attach_parent_console();
            let _ = e.print();
            return if e.use_stderr() { ExitCode::FAILURE } else { ExitCode::SUCCESS };
        }
    };
    let r = match cli.cmd {
        None => match installed_here() {
            Some(dir) => launch::run(&dir, true),
            None => gui::run(gui::Start::Install(None)),
        },
        Some(Cmd::Launch { dir, no_browser }) => match dir.or_else(installed_here) {
            Some(d) => launch::run(&d, !no_browser),
            None => Err(anyhow::anyhow!("no inkling.toml next to this program; pass --dir <install folder>")),
        },
        Some(Cmd::Setup { dir }) => gui::run(gui::Start::Install(dir.or_else(installed_here))),
        Some(Cmd::Detect { json }) => detect(json),
        Some(Cmd::Install(a)) => install_cli(a),
        Some(Cmd::Command { dir }) => command(dir),
        Some(Cmd::Uninstall { dir, yes, keep_models }) => {
            let dir = dir.or_else(installed_here).unwrap_or_else(install::default_dir);
            if yes {
                install::uninstall(&dir, keep_models, &progress::ConsoleReporter::new())
            } else {
                gui::run(gui::Start::Uninstall(dir))
            }
        }
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn detect(json: bool) -> anyhow::Result<()> {
    let hw = hw::detect();
    let a = plan::assess(&hw);
    let tc = engine::Toolchain::detect();
    if json {
        let v = serde_json::json!({
            "version": release::VERSION,
            "repo": release::REPO,
            "hardware": hw,
            "assessment": a,
            "toolchain": { "visual_studio": tc.vs, "cuda_path": tc.cuda_path, "nvcc": tc.nvcc_version, "missing": tc.missing() },
            "installed_dir": system::installed_dir(),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!("Inkling {} hardware check\n", release::VERSION);
    if hw.gpus.is_empty() {
        println!("GPU       none ({})", hw.gpu_error.clone().unwrap_or_default());
    }
    for g in &hw.gpus {
        println!(
            "GPU {}     {}  {:.1} GB  {}  driver {}{}",
            g.index,
            g.name,
            g.vram_gib(),
            g.sm(),
            g.driver,
            hw.driver_cuda.map(|(a, b)| format!(" (CUDA {a}.{b})")).unwrap_or_default()
        );
    }
    println!("RAM       {:.1} GB", hw.ram_gib());
    println!(
        "CPU       {}  {} cores / {} threads{}",
        hw.cpu_name,
        hw.physical_cores,
        hw.logical_cores,
        if hw.avx2 { "  AVX2" } else { "  no AVX2" }
    );
    println!("\nVerdict   {}: {}", a.support.label(), a.summary);
    for n in &a.notes {
        println!("          - {n}");
    }
    let s = &a.settings;
    println!(
        "\nSettings  vram_cap_gib {:.1} · cpu_threads {} · max_ctx {} (fits ~{}k) · vision {}",
        s.vram_cap_gib,
        s.cpu_threads,
        s.max_ctx,
        a.max_ctx_fit / 1000,
        s.vision.name()
    );
    println!(
        "Engine    {}",
        if a.prebuilt {
            "prebuilt (sm_89 + sm_120a)".to_string()
        } else {
            format!("build from source for {}", a.cuda_arch.clone().unwrap_or("?".into()))
        }
    );
    let missing = tc.missing();
    println!(
        "Toolchain {}",
        if missing.is_empty() {
            format!("ready for a source build (CUDA {})", tc.nvcc_version.unwrap_or_default())
        } else {
            format!("a source build would need: {}", missing.join("; "))
        }
    );
    Ok(())
}

fn install_cli(a: InstallArgs) -> anyhow::Result<()> {
    let hw = hw::detect();
    let assessment = plan::assess(&hw);
    println!("{}: {}", assessment.support.label(), assessment.summary);
    for n in &assessment.notes {
        println!("  - {n}");
    }
    if assessment.support == plan::Support::Unsupported && !a.force {
        anyhow::bail!("not installing on unsupported hardware (pass --force to try anyway; see AGENTS.md)");
    }
    let dir = a.dir.clone().unwrap_or_else(install::default_dir);
    let mut cfg = base_config(&dir, &hw, &assessment);
    let s = &mut cfg.server;
    if let Some(v) = a.max_ctx {
        s.max_ctx = v;
    }
    if let Some(v) = a.vram_cap_gib {
        s.vram_cap_gib = v;
    }
    if let Some(v) = a.cpu_threads {
        s.cpu_threads = v;
    }
    if let Some(v) = a.port {
        s.port = v;
    }
    if let Some(v) = &a.vision {
        s.vision = match v.as_str() {
            "cpu" => Vision::Cpu,
            "gpu" => Vision::Gpu,
            "off" => Vision::Off,
            _ => anyhow::bail!("--vision must be cpu, gpu or off"),
        };
    }
    for kv in &a.env {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("--env expects KEY=VALUE, got {kv}"))?;
        cfg.env.insert(k.trim().into(), v.trim().into());
    }
    let engine = if a.from_source || a.source.is_some() {
        let arch = a.arch.clone().or(assessment.cuda_arch.clone()).unwrap_or_else(|| "89".into());
        EngineSource::Source { dir: a.source.clone(), arch }
    } else {
        if !assessment.prebuilt && a.engine_from.is_none() {
            anyhow::bail!(
                "the prebuilt engine does not cover this GPU; use --from-source (needs Visual Studio 2022 + CUDA 13)"
            );
        }
        EngineSource::Prebuilt { from: a.engine_from.clone() }
    };
    println!("Installing to {}\n", dir.display());
    let o = install::Options {
        dir: dir.clone(),
        config: cfg,
        engine,
        models_from: a.models_from,
        desktop_shortcut: !a.no_desktop_shortcut,
        start_menu: !a.no_start_menu,
        skip_models: a.skip_models,
    };
    install::run(&o, &progress::ConsoleReporter::new())?;
    println!("\nInstalled. Start it from the Start menu (Inkling), or: \"{}\"", dir.join("Inkling.exe").display());
    Ok(())
}

/// The settings to start from: the existing inkling.toml (keeps [env] and extra_args), else the recommendation.
pub fn base_config(dir: &Path, hw: &hw::Hardware, a: &plan::Assessment) -> Config {
    let mut cfg = Config::load(dir).unwrap_or_else(|_| Config { server: a.settings.clone(), ..Config::default() });
    if let Some(g) = &a.gpu {
        cfg.install.gpu = format!("{} ({:.1} GB, {})", g.name, g.vram_gib(), g.sm());
        if hw.gpus.len() > 1 {
            cfg.env.insert("CUDA_DEVICE_ORDER".into(), "PCI_BUS_ID".into());
            cfg.env.insert("CUDA_VISIBLE_DEVICES".into(), g.index.to_string());
        }
    }
    cfg.install.detected = format!(
        "{:.0} GB RAM, {} ({} cores), {}",
        hw.ram_gib(),
        hw.cpu_name,
        hw.physical_cores,
        a.support.label()
    );
    cfg
}

fn command(dir: Option<PathBuf>) -> anyhow::Result<()> {
    let dir = dir.or_else(installed_here).or_else(system::installed_dir).ok_or_else(|| anyhow::anyhow!("pass --dir"))?;
    println!("{}", Config::load(&dir)?.command_line(&dir));
    Ok(())
}
