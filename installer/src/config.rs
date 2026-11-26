//! inkling.toml: the launcher's settings, written by the installer for this machine and read at every start.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const FILE: &str = "inkling.toml";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Vision {
    /// image tower on the CPU: no VRAM, images take a few seconds longer
    Cpu,
    /// image tower on the GPU (~0.6 GB of VRAM)
    Gpu,
    /// text only
    Off,
}

impl Vision {
    pub fn label(self) -> &'static str {
        match self {
            Vision::Cpu => "Images on the CPU (no VRAM)",
            Vision::Gpu => "Images on the GPU (0.6 GB VRAM)",
            Vision::Off => "Text only",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Vision::Cpu => "cpu",
            Vision::Gpu => "gpu",
            Vision::Off => "off",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Server {
    pub host: String,
    pub port: u16,
    pub max_ctx: u32,
    pub vram_cap_gib: f64,
    pub cpu_threads: u32,
    pub vision: Vision,
    pub vision_threads: u32,
    pub image_max_tokens: u32,
    pub spec: bool,
    /// remove the model's refusals (the abliteration file); false runs the original model
    pub abliterated: bool,
    pub extra_args: Vec<String>,
}

impl Default for Server {
    /// the reference machine (RTX 4070 SUPER 12 GB, Ryzen 7 5800X3D, 32 GB DDR4)
    fn default() -> Self {
        Server {
            host: "127.0.0.1".into(),
            port: 8003,
            max_ctx: 600_064,
            vram_cap_gib: 11.3,
            cpu_threads: 7,
            vision: Vision::Cpu,
            vision_threads: 8,
            image_max_tokens: 1024,
            spec: true,
            abliterated: true,
            extra_args: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct InstallInfo {
    pub version: String,
    /// "prebuilt sm_89+sm_120a" or "source <arch>"
    pub engine: String,
    pub gpu: String,
    pub detected: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: Server,
    /// environment of the server process: QUILL_* switches (AGENTS.md), CUDA_VISIBLE_DEVICES
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub install: InstallInfo,
}

impl Default for Config {
    fn default() -> Self {
        Config { server: Server::default(), env: default_env(), install: InstallInfo::default() }
    }
}

pub fn default_env() -> BTreeMap<String, String> {
    // Inkling-2-Mini is trained for 3 loop passes; other counts garble its output
    BTreeMap::from([("QUILL_PASSES".to_string(), "3".to_string())])
}

impl Config {
    pub fn path(dir: &Path) -> PathBuf {
        dir.join(FILE)
    }

    pub fn exists(dir: &Path) -> bool {
        Self::path(dir).is_file()
    }

    pub fn load(dir: &Path) -> Result<Config> {
        let p = Self::path(dir);
        let t = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        toml::from_str(&t).with_context(|| format!("parsing {}", p.display()))
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let p = Self::path(dir);
        std::fs::write(&p, self.to_commented_toml()).with_context(|| format!("writing {}", p.display()))
    }

    /// The file with a comment on every setting (serde would drop them).
    pub fn to_commented_toml(&self) -> String {
        let s = &self.server;
        let v = |x: toml::Value| x.to_string();
        let mut out = String::new();
        out += "# Inkling launcher settings, written by the installer for this machine.\n";
        out += "# Edit, then restart Inkling (close the server window, start it again). AGENTS.md explains each setting.\n\n";
        out += "[server]\n";
        out += &format!("host = {}\n", v(s.host.clone().into()));
        out += &format!("port = {}\n", s.port);
        out += "# context window in tokens; unused context costs nothing until a chat fills it (KV ~9 KiB per token)\n";
        out += &format!("max_ctx = {}\n", s.max_ctx);
        out += "# GPU memory in use by ALL programs (Task Manager's number) that the engine stays under; card size - ~0.7 GB\n";
        out += &format!("vram_cap_gib = {:.1}\n", s.vram_cap_gib);
        out += "# CPU workers for the experts that do not fit on the GPU: physical cores - 1\n";
        out += &format!("cpu_threads = {}\n", s.cpu_threads);
        out += "# images: \"cpu\" (no VRAM), \"gpu\" (0.6 GB VRAM, faster) or \"off\"\n";
        out += &format!("vision = {}\n", v(s.vision.name().into()));
        out += &format!("vision_threads = {}\n", s.vision_threads);
        out += &format!("image_max_tokens = {}\n", s.image_max_tokens);
        out += "# DFlash2 speculative decoding (most of the speed)\n";
        out += &format!("spec = {}\n", s.spec);
        out += "# true: refusals removed (abliteration); false: the original model. Turning it on needs the installer
";
        out += "# to have run with it on once, which downloads the file
";
        out += &format!("abliterated = {}
", s.abliterated);
        out += "# more ingine-serve flags, e.g. [\"--verify-widths\", \"2,4,8\"] (ingine-serve --help)\n";
        out += &format!(
            "extra_args = [{}]\n\n",
            s.extra_args.iter().map(|a| v(a.clone().into())).collect::<Vec<_>>().join(", ")
        );
        out += "# environment of the server: QUILL_* switches (AGENTS.md). QUILL_PASSES must stay 3 for Inkling-2-Mini.\n";
        out += "[env]\n";
        for (k, val) in &self.env {
            out += &format!("{} = {}\n", bare_or_quoted(k), v(val.clone().into()));
        }
        out += "\n# what the installer found (informational)\n[install]\n";
        let i = &self.install;
        for (k, val) in [("version", &i.version), ("engine", &i.engine), ("gpu", &i.gpu), ("detected", &i.detected)] {
            out += &format!("{k} = {}\n", v(val.clone().into()));
        }
        out
    }

    /// the id the server reports on its API
    pub fn model_id(&self) -> &'static str {
        if self.server.abliterated { "inkling-2-mini-abliterated" } else { "inkling-2-mini" }
    }

    pub fn url(&self) -> String {
        let h = match self.server.host.as_str() {
            "0.0.0.0" | "::" | "" => "127.0.0.1",
            h => h,
        };
        format!("http://{h}:{}/", self.server.port)
    }

    pub fn server_args(&self, dir: &Path) -> Vec<String> {
        let s = &self.server;
        let m = dir.join("models");
        let p = |x: PathBuf| x.to_string_lossy().into_owned();
        let mut a: Vec<String> = vec![
            "--engine".into(),
            "--model-dir".into(),
            p(m.join("Inkling-2-Mini")),
        ];
        if s.abliterated {
            a.extend(["--ablation".into(), p(m.join("ablation-ingine").join("inkling-2-mini-abliterated.safetensors"))]);
        }
        if s.spec {
            a.extend(["--spec".into(), "--draft-dir".into(), p(m.join("Inkling-2-Mini-DFlash2"))]);
        }
        a.extend([
            "--max-ctx".into(),
            s.max_ctx.to_string(),
            "--dense-fp8".into(),
            "--vram-cap-gib".into(),
            format!("{:.2}", s.vram_cap_gib),
            "--cpu-threads".into(),
            s.cpu_threads.to_string(),
        ]);
        match s.vision {
            Vision::Off => a.push("--no-vision".into()),
            Vision::Cpu => a.extend([
                "--vision-cpu".into(),
                "--vision-threads".into(),
                s.vision_threads.to_string(),
                "--image-max-tokens".into(),
                s.image_max_tokens.to_string(),
            ]),
            Vision::Gpu => a.extend(["--image-max-tokens".into(), s.image_max_tokens.to_string()]),
        }
        a.extend([
            "--ui".into(),
            p(dir.join("app").join("chat.html")),
            "--host".into(),
            s.host.clone(),
            "--port".into(),
            s.port.to_string(),
        ]);
        a.extend(s.extra_args.iter().cloned());
        a
    }

    pub fn server_exe(dir: &Path) -> PathBuf {
        dir.join("bin").join("ingine-serve.exe")
    }

    pub fn server_command(&self, dir: &Path) -> Command {
        let mut c = Command::new(Self::server_exe(dir));
        c.args(self.server_args(dir)).current_dir(dir).envs(&self.env);
        c
    }

    /// The command as one line, quoted for cmd.exe / PowerShell.
    pub fn command_line(&self, dir: &Path) -> String {
        let q = |s: &str| if s.contains(' ') || s.is_empty() { format!("\"{s}\"") } else { s.to_string() };
        let mut out = String::new();
        for (k, v) in &self.env {
            out += &format!("set {k}={v}\n");
        }
        out += &q(&Self::server_exe(dir).to_string_lossy());
        for a in self.server_args(dir) {
            out.push(' ');
            out += &q(&a);
        }
        out
    }
}

fn bare_or_quoted(k: &str) -> String {
    if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        k.to_string()
    } else {
        toml::Value::from(k.to_string()).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commented_toml_round_trips() {
        let mut c = Config::default();
        c.server.extra_args = vec!["--verify-widths".into(), "2,4,8".into()];
        c.env.insert("QUILL_ASSIST".into(), "0.2".into());
        c.install.gpu = "NVIDIA GeForce RTX 4070 SUPER".into();
        let back: Config = toml::from_str(&c.to_commented_toml()).unwrap();
        assert_eq!(back.server, c.server);
        assert_eq!(back.env, c.env);
        assert_eq!(back.install.gpu, c.install.gpu);
    }

    #[test]
    fn original_model_has_no_ablation() {
        let mut c = Config::default();
        c.server.abliterated = false;
        assert!(!c.server_args(Path::new(r"C:\Inkling")).contains(&"--ablation".to_string()));
        assert_eq!(c.model_id(), "inkling-2-mini");
        let back: Config = toml::from_str(&c.to_commented_toml()).unwrap();
        assert!(!back.server.abliterated);
    }

    #[test]
    fn reference_command_matches_the_original_launcher() {
        let c = Config::default();
        let a = c.server_args(Path::new(r"C:\Inkling")).join(" ");
        for want in [
            "--engine",
            r"--ablation C:\Inkling\models\ablation-ingine\inkling-2-mini-abliterated.safetensors",
            r"--spec --draft-dir C:\Inkling\models\Inkling-2-Mini-DFlash2",
            "--max-ctx 600064 --dense-fp8",
            "--vision-cpu",
            "--image-max-tokens 1024",
            r"--ui C:\Inkling\app\chat.html --host 127.0.0.1 --port 8003",
        ] {
            assert!(a.contains(want), "missing {want:?} in {a}");
        }
        assert_eq!(c.env.get("QUILL_PASSES").map(String::as_str), Some("3"));
    }
}
