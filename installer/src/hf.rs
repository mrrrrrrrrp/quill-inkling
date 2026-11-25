//! Model downloads from Hugging Face at pinned revisions (no login needed; HF_TOKEN is used when set).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::download::{self, Check, Http};
use crate::progress::Reporter;
use crate::release::{self, ModelRepo};
use crate::util::fmt_bytes;

#[derive(Clone, Debug)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    pub check: Check,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    oid: String,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
    size: u64,
}

pub fn list_files(http: &Http, m: &ModelRepo, rev: &str) -> Result<Vec<RepoFile>> {
    let mut url = Some(format!("https://huggingface.co/api/models/{}/tree/{rev}?recursive=true", m.repo));
    let mut out = Vec::new();
    while let Some(u) = url {
        let (body, next) = http.get_text(&u)?;
        let entries: Vec<TreeEntry> = serde_json::from_str(&body).with_context(|| format!("parsing the file list of {}", m.repo))?;
        for e in entries.into_iter().filter(|e| e.kind == "file") {
            let wanted = m.include.is_empty()
                || m.include.iter().any(|p| if p.ends_with('/') { e.path.starts_with(p) } else { e.path == *p });
            if !wanted {
                continue;
            }
            let (size, check) = match e.lfs {
                Some(l) => (l.size, Check::Sha256(l.oid)),
                None => (e.size, Check::GitBlob(e.oid)),
            };
            out.push(RepoFile { path: e.path, size, check });
        }
        url = next;
    }
    if out.is_empty() {
        bail!("{} @ {rev}: no files listed", m.repo);
    }
    Ok(out)
}

pub fn file_url(repo: &str, rev: &str, path: &str) -> String {
    let enc: Vec<String> = path.split('/').map(encode_segment).collect();
    format!("https://huggingface.co/{repo}/resolve/{rev}/{}", enc.join("/"))
}

fn encode_segment(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Files already checked: path -> (size, mtime, checksum), so a re-run does not hash 20 GB again.
#[derive(Default, Serialize, Deserialize)]
struct Verified(BTreeMap<String, (u64, u64, String)>);

const VERIFIED: &str = ".inkling-verified.json";

fn mtime(p: &Path) -> u64 {
    p.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0)
}

fn check_id(c: &Check) -> String {
    match c {
        Check::Sha256(h) => format!("sha256:{h}"),
        Check::GitBlob(h) => format!("git:{h}"),
        Check::None => String::new(),
    }
}

/// Total bytes the models need, and how many of them are still to download (for the UI).
pub fn plan_bytes(http: &Http, models_dir: &Path) -> Result<(u64, u64)> {
    let (mut total, mut missing) = (0, 0);
    for m in release::MODELS {
        let rev = release::pinned_revision(m.repo).context("model not pinned")?;
        for f in list_files(http, m, rev)? {
            total += f.size;
            let p = models_dir.join(m.dir).join(&f.path);
            if p.metadata().map(|md| md.len() != f.size).unwrap_or(true) {
                missing += f.size;
            }
        }
    }
    Ok((total, missing))
}

/// Brings `models/<dir>` to the pinned revision: verified files are kept, files from `models_from` are linked
/// or copied, the rest is downloaded.
pub fn sync_repo(http: &Http, m: &ModelRepo, models_dir: &Path, models_from: Option<&Path>, rep: &dyn Reporter) -> Result<()> {
    let rev = release::pinned_revision(m.repo).with_context(|| format!("{} is not in models/PINS.txt", m.repo))?;
    let dir = models_dir.join(m.dir);
    std::fs::create_dir_all(&dir)?;
    let files = list_files(http, m, rev)?;
    let total: u64 = files.iter().map(|f| f.size).sum();
    rep.log(&format!("{} @ {} ({} files, {})", m.repo, &rev[..7], files.len(), fmt_bytes(total)));

    let vpath = dir.join(VERIFIED);
    let mut verified: Verified = std::fs::read_to_string(&vpath).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let save = |v: &Verified| std::fs::write(&vpath, serde_json::to_string_pretty(v).unwrap_or_default());

    for f in &files {
        let dest = dir.join(f.path.replace('/', "\\"));
        let id = check_id(&f.check);
        let label = format!("{}/{}", m.dir, f.path);
        // already verified and unchanged
        if let Some((size, mt, cid)) = verified.0.get(&f.path) {
            if *size == f.size && *cid == id && dest.metadata().map(|md| md.len() == f.size).unwrap_or(false) && *mt == mtime(&dest) {
                continue;
            }
        }
        // an existing copy (earlier install, or --models-from)
        if !dest.exists() {
            if let Some(src) = models_from.map(|d| source_in(d, m.dir, &f.path)) {
                if src.metadata().map(|md| md.len() == f.size).unwrap_or(false) {
                    if let Some(d) = dest.parent() {
                        std::fs::create_dir_all(d)?;
                    }
                    if std::fs::hard_link(&src, &dest).is_ok() {
                        rep.log(&format!("{label}: linked from {}", src.display()));
                    } else {
                        rep.log(&format!("{label}: copying from {}", src.display()));
                        std::fs::copy(&src, &dest).with_context(|| format!("copying {}", src.display()))?;
                    }
                }
            }
        }
        let mut ok = false;
        if dest.metadata().map(|md| md.len() == f.size).unwrap_or(false) {
            match download::verify(&dest, &f.check, &label, rep) {
                Ok(()) => ok = true,
                Err(e) => rep.log(&format!("{e:#}")),
            }
        }
        if !ok {
            if rep.cancelled() {
                return Err(crate::progress::Cancelled.into());
            }
            http.download(&file_url(m.repo, rev, &f.path), &dest, Some(f.size), &f.check, &label, rep)?;
        }
        verified.0.insert(f.path.clone(), (f.size, mtime(&dest), id));
        save(&verified)?;
    }
    Ok(())
}

/// `models_from` may be a models folder (holding <dir>\...) or the repo folder itself.
fn source_in(from: &Path, dir: &str, path: &str) -> PathBuf {
    let nested = from.join(dir);
    let base = if nested.is_dir() { nested } else { from.to_path_buf() };
    base.join(path.replace('/', "\\"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Network test (~1 GB): a chunked download from Hugging Face, cancelled part-way, then resumed and verified.
    /// cargo test --release -- --ignored hf_download_resumes
    #[test]
    #[ignore]
    fn hf_download_resumes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Stop(u64, AtomicBool, std::sync::Mutex<Vec<String>>);
        impl Reporter for Stop {
            fn step(&self, _: &str) {}
            fn progress(&self, _: &str, done: u64, _: u64) {
                if done >= self.0 {
                    self.1.store(true, Ordering::Relaxed);
                }
            }
            fn log(&self, l: &str) {
                self.2.lock().unwrap().push(l.to_string());
            }
            fn cancelled(&self) -> bool {
                self.1.load(Ordering::Relaxed)
            }
        }
        let http = Http::new();
        let m = &release::MODELS[1];
        let rev = release::pinned_revision(m.repo).unwrap();
        let f = list_files(&http, m, rev).unwrap().into_iter().find(|f| f.path == "model.safetensors").unwrap();
        let dir = std::env::temp_dir().join(format!("inkling-dl-{}", std::process::id()));
        let dest = dir.join("model.safetensors");
        let url = file_url(m.repo, rev, &f.path);
        // past the first round of WORKERS x 32 MiB chunks, so some are complete when it stops
        let first = Stop(700 << 20, AtomicBool::new(false), Default::default());
        let r = http.download(&url, &dest, Some(f.size), &f.check, "dflash2", &first);
        assert!(r.unwrap_err().is::<crate::progress::Cancelled>());
        let second = Stop(u64::MAX, AtomicBool::new(false), Default::default());
        http.download(&url, &dest, Some(f.size), &f.check, "dflash2", &second).unwrap();
        let logs = second.2.lock().unwrap().join("\n");
        assert!(logs.contains("resuming"), "{logs}");
        assert_eq!(dest.metadata().unwrap().len(), f.size);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Network speed test (~1 GB): cargo test --release -- --ignored hf_download_speed --nocapture
    #[test]
    #[ignore]
    fn hf_download_speed() {
        struct Quiet;
        impl Reporter for Quiet {
            fn step(&self, _: &str) {}
            fn progress(&self, _: &str, _: u64, _: u64) {}
            fn log(&self, l: &str) {
                println!("{l}");
            }
        }
        let http = Http::new();
        let m = &release::MODELS[1];
        let rev = release::pinned_revision(m.repo).unwrap();
        let f = list_files(&http, m, rev).unwrap().into_iter().find(|f| f.path == "model.safetensors").unwrap();
        let dir = std::env::temp_dir().join(format!("inkling-speed-{}", std::process::id()));
        let dest = dir.join("model.safetensors");
        let t = std::time::Instant::now();
        http.download(&file_url(m.repo, rev, &f.path), &dest, Some(f.size), &Check::None, "dflash2", &Quiet).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!("SPEED {:.0} MB in {s:.1} s = {:.0} MB/s", f.size as f64 / 1e6, f.size as f64 / 1e6 / s);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn urls_are_encoded_per_segment() {
        assert_eq!(
            file_url("a/b", "abc", "tiktoken/x y.json"),
            "https://huggingface.co/a/b/resolve/abc/tiktoken/x%20y.json"
        );
    }
}
