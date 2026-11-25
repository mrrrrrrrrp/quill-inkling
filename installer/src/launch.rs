//! The launcher: starts ingine-serve in this console window, opens the chat page once /health answers, and stops
//! the server when the window closes.

use anyhow::{bail, Result};
use std::io::BufRead;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::system::{self, Job};
use crate::util::shell_open;

pub fn healthy(cfg: &Config) -> bool {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(2)))
        .http_status_as_error(false)
        .build()
        .into();
    agent.get(&format!("{}health", cfg.url())).call().map(|r| r.status().as_u16() == 200).unwrap_or(false)
}

pub fn run(dir: &Path, open_browser: bool) -> Result<()> {
    if system::started_with_console() {
        // inkling-cli.exe in a terminal: the server runs in it; Ctrl+C stops both
        println!("Inkling server: Ctrl+C stops it.");
    } else {
        system::new_console("Inkling server - close this window to stop");
    }
    let r = run_in_console(dir, open_browser);
    if let Err(e) = &r {
        if !system::started_with_console() {
            // our own window: keep it open so the log can be read
            println!("\n{e:#}");
            wait_enter();
        }
    }
    r
}

fn run_in_console(dir: &Path, open_browser: bool) -> Result<()> {
    let cfg = Config::load(dir)?;
    let url = cfg.url();
    if healthy(&cfg) {
        println!("Inkling is already running: {url}");
        if open_browser {
            shell_open(&url);
        }
        std::thread::sleep(Duration::from_secs(2));
        return Ok(());
    }
    let exe = Config::server_exe(dir);
    if !exe.is_file() {
        bail!("{} is missing. Run \"Inkling Settings\" from the Start menu and choose Repair.", exe.display());
    }
    let s = &cfg.server;
    println!("Inkling 2 Mini on quill  ·  {}", dir.display());
    println!(
        "context {}k · VRAM cap {:.1} GB · {} CPU threads · images: {} · speculative decoding {}",
        s.max_ctx / 1000,
        s.vram_cap_gib,
        s.cpu_threads,
        s.vision.name(),
        if s.spec { "on" } else { "off" }
    );
    println!("Loading the model takes about a minute...\n");

    let mut child = cfg.server_command(dir).spawn()?;
    let _job = Job::kill_on_close(&child);
    let t0 = Instant::now();
    loop {
        if let Some(st) = child.try_wait()? {
            return stopped(st.code());
        }
        if healthy(&cfg) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1000));
    }
    println!("\n=================================================================================");
    println!(" Ready after {} s.   Chat: {url}", t0.elapsed().as_secs());
    println!(" OpenAI-compatible API: {url}v1   (model id: inkling-2-mini-abliterated)");
    println!(" Close this window to stop the server and free the GPU.");
    println!("=================================================================================\n");
    if open_browser {
        shell_open(&url);
    }
    let st = child.wait()?;
    if st.success() {
        return Ok(());
    }
    stopped(st.code())
}

fn stopped(code: Option<i32>) -> Result<()> {
    println!("\nThe server stopped (exit code {}). The lines above say why. Common fixes, in inkling.toml:", code.map(|c| c.to_string()).unwrap_or("?".into()));
    println!("  - \"cannot hold the KV cache of max_ctx\": lower max_ctx (e.g. 262144).");
    println!("  - out of memory on the GPU: close other GPU programs, or lower vram_cap_gib a little.");
    println!("  - host memory / cudaHostRegister errors: close programs to free RAM (the engine pins ~16 GB).");
    println!("  - address already in use: another program uses the port; change port.");
    println!("AGENTS.md (in the repository) explains every setting.");
    bail!("server exited")
}

fn wait_enter() {
    println!("Press Enter to close this window.");
    let _ = std::io::stdin().lock().read_line(&mut String::new());
}
