//! HTTP downloads: parallel byte ranges for large files (resumable: finished chunks are recorded next to the
//! .part file), one stream for small ones; every file is verified (SHA-256, or Git's blob SHA-1) before it is
//! renamed into place.

use anyhow::{anyhow, bail, Context, Result};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use crate::progress::{Cancelled, Reporter};
use crate::util::hex;

// 32 MiB chunks (the .part.done files of earlier runs count in these units: keep it), 16 parallel connections.
// Measured on a 1 Gbit line against Hugging Face's CDN: 1 connection 55 MB/s, 16 connections ~90-115 MB/s.
const CHUNK: u64 = 32 << 20;
const WORKERS: usize = 16;
const RETRIES: u32 = 6;

#[derive(Clone, Debug)]
pub enum Check {
    Sha256(String),
    /// Git's object id: SHA-1 of "blob <size>\0" + content (Hugging Face reports it for small files)
    GitBlob(String),
    None,
}

pub struct Http {
    agent: ureq::Agent,
    hf_token: Option<String>,
}

impl Http {
    pub fn new() -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .timeout_per_read(Some(Duration::from_secs(60)))
            // keep every worker's connection open between chunks: a new TLS connection per chunk restarts TCP's
            // slow start, and 32 MiB never gets up to speed (ureq keeps 3 per host by default)
            .max_idle_connections(4 * WORKERS)
            .max_idle_connections_per_host(2 * WORKERS)
            .http_status_as_error(false)
            .user_agent(concat!("inkling-installer/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        let hf_token = std::env::var("HF_TOKEN").ok().filter(|t| !t.trim().is_empty());
        Http { agent, hf_token }
    }

    fn get(&self, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
        let r = self.agent.get(url);
        match &self.hf_token {
            // ureq drops Authorization on redirects to other hosts (the CDN)
            Some(t) if url.starts_with("https://huggingface.co/") => r.header("Authorization", format!("Bearer {t}")),
            _ => r,
        }
    }

    /// Follows redirects once and returns the final URL (Hugging Face's resolve URL -> a signed CDN URL), so the
    /// chunks go straight to the CDN instead of through a redirect each.
    fn resolve(&self, url: &str) -> Result<String> {
        let mut u = url.to_string();
        for _ in 0..8 {
            let resp = self.get(&u).config().max_redirects(0).build().header("Range", "bytes=0-0").call()?;
            let status = resp.status().as_u16();
            if !(300..400).contains(&status) {
                return Ok(u);
            }
            let loc = resp.headers().get("location").and_then(|v| v.to_str().ok()).context("redirect without a Location")?;
            u = if loc.starts_with('/') {
                let origin_end = u.find("://").map(|i| i + 3).and_then(|i| u[i..].find('/').map(|j| i + j)).unwrap_or(u.len());
                format!("{}{loc}", &u[..origin_end])
            } else {
                loc.to_string()
            };
        }
        bail!("too many redirects for {url}")
    }

    /// GET a text body; also returns the URL of the next page from a `Link: <...>; rel="next"` header.
    pub fn get_text(&self, url: &str) -> Result<(String, Option<String>)> {
        let mut resp = self.get(url).call().with_context(|| format!("GET {url}"))?;
        let status = resp.status().as_u16();
        let next = resp.headers().get("link").and_then(|v| v.to_str().ok()).and_then(next_link);
        let body = resp.body_mut().with_config().limit(64 << 20).read_to_string().unwrap_or_default();
        if status != 200 {
            bail!("GET {url}: HTTP {status} {}", body.chars().take(200).collect::<String>());
        }
        Ok((body, next))
    }

    /// Downloads `url` to `dest` (size in bytes if known), verifying it. Resumes an earlier partial download.
    pub fn download(
        &self,
        url: &str,
        dest: &Path,
        size: Option<u64>,
        check: &Check,
        label: &str,
        rep: &dyn Reporter,
    ) -> Result<()> {
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let part = suffixed(dest, ".part");
        match size {
            Some(n) if n > 2 * CHUNK => self.download_chunked(url, &part, n, label, rep)?,
            _ => self.download_stream(url, &part, size, label, rep)?,
        }
        verify(&part, check, label, rep)?;
        if dest.exists() {
            std::fs::remove_file(dest).with_context(|| format!("replacing {}", dest.display()))?;
        }
        std::fs::rename(&part, dest).with_context(|| format!("moving {} into place", dest.display()))?;
        let _ = std::fs::remove_file(suffixed(&part, ".done"));
        Ok(())
    }

    fn download_stream(&self, url: &str, part: &Path, size: Option<u64>, label: &str, rep: &dyn Reporter) -> Result<()> {
        let mut attempt = 0;
        loop {
            match self.try_stream(url, part, size, label, rep) {
                Ok(()) => return Ok(()),
                Err(e) if e.is::<Cancelled>() => return Err(e),
                Err(e) if attempt < RETRIES => {
                    rep.log(&format!("{label}: {e:#}; retrying"));
                    attempt += 1;
                    std::thread::sleep(Duration::from_secs(2u64 << attempt.min(4)));
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn try_stream(&self, url: &str, part: &Path, size: Option<u64>, label: &str, rep: &dyn Reporter) -> Result<()> {
        let resp = self.get(url).call().with_context(|| format!("GET {url}"))?;
        let status = resp.status().as_u16();
        if status != 200 {
            bail!("GET {url}: HTTP {status}");
        }
        let total = size
            .or_else(|| resp.headers().get("content-length").and_then(|v| v.to_str().ok()?.parse().ok()))
            .unwrap_or(0);
        let mut reader = resp.into_body().into_reader();
        let mut f = File::create(part).with_context(|| format!("creating {}", part.display()))?;
        let mut buf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        loop {
            if rep.cancelled() {
                return Err(Cancelled.into());
            }
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n])?;
            done += n as u64;
            rep.progress(label, done, total);
        }
        if total > 0 && done != total {
            bail!("{label}: got {done} of {total} bytes");
        }
        Ok(())
    }

    fn download_chunked(&self, url: &str, part: &Path, size: u64, label: &str, rep: &dyn Reporter) -> Result<()> {
        let state_path = suffixed(part, ".done");
        let n = size.div_ceil(CHUNK);
        let chunk_len = |i: u64| (size - i * CHUNK).min(CHUNK);
        // chunks finished by an earlier run (only while the .part file is the right size)
        let mut done: HashSet<u64> = HashSet::new();
        if part.metadata().map(|m| m.len() == size).unwrap_or(false) {
            if let Ok(f) = File::open(&state_path) {
                done = BufReader::new(f).lines().map_while(|l| l.ok()).filter_map(|l| l.trim().parse().ok()).filter(|&i| i < n).collect();
            }
        } else {
            let _ = std::fs::remove_file(&state_path);
        }
        if !done.is_empty() {
            rep.log(&format!("{label}: resuming ({} of {n} chunks already downloaded)", done.len()));
        }
        let f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(part)
            .with_context(|| format!("opening {}", part.display()))?;
        // Sparse, so a chunk written far past what has been written so far does not make NTFS zero-fill the gap
        // first (measured: 16 workers scattering writes into a plain preallocated file ran at ~20 MB/s on a
        // 1 Gbit line that delivers 100+ MB/s to memory).
        set_sparse(&f, true);
        f.set_len(size)?;
        let state = Mutex::new(OpenOptions::new().create(true).append(true).open(&state_path)?);
        let queue: Mutex<Vec<u64>> = Mutex::new((0..n).rev().filter(|i| !done.contains(i)).collect());
        let bytes = AtomicU64::new(done.iter().map(|&i| chunk_len(i)).sum());
        let failure: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        let active = AtomicUsize::new(WORKERS);
        // the CDN URL behind the redirect, shared by the workers; re-resolved when its signature expires
        let target = Mutex::new(self.resolve(url).unwrap_or_else(|_| url.to_string()));

        std::thread::scope(|s| {
            for _ in 0..WORKERS {
                s.spawn(|| {
                    // each worker writes through its own handle: synchronous I/O on one shared handle is serialised
                    let fw = match OpenOptions::new().write(true).open(part) {
                        Ok(h) => h,
                        Err(e) => {
                            *failure.lock().unwrap() = Some(anyhow::Error::from(e).context(format!("opening {}", part.display())));
                            active.fetch_sub(1, Ordering::SeqCst);
                            return;
                        }
                    };
                    let f = &fw;
                    loop {
                        if rep.cancelled() || failure.lock().unwrap().is_some() {
                            break;
                        }
                        let Some(i) = queue.lock().unwrap().pop() else { break };
                        let (a, len) = (i * CHUNK, chunk_len(i));
                        let mut attempt = 0;
                        loop {
                            let u = target.lock().unwrap().clone();
                            match self.fetch_range(&u, f, a, len, &bytes, rep) {
                                Ok(()) => {
                                    let _ = writeln!(state.lock().unwrap(), "{i}");
                                    break;
                                }
                                Err(e) if e.is::<Cancelled>() => break,
                                Err(e) if attempt < RETRIES => {
                                    attempt += 1;
                                    if expired(&e) {
                                        if let Ok(n) = self.resolve(url) {
                                            *target.lock().unwrap() = n;
                                        }
                                        continue;
                                    }
                                    if attempt >= 2 {
                                        rep.log(&format!("{label}: {e:#}; retrying"));
                                    }
                                    std::thread::sleep(Duration::from_secs(1u64 << attempt.min(5)));
                                }
                                Err(e) => {
                                    *failure.lock().unwrap() = Some(e);
                                    break;
                                }
                            }
                        }
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
            while active.load(Ordering::SeqCst) > 0 {
                rep.progress(label, bytes.load(Ordering::Relaxed), size);
                std::thread::sleep(Duration::from_millis(150));
            }
            rep.progress(label, bytes.load(Ordering::Relaxed), size);
        });
        if let Some(e) = failure.into_inner().unwrap() {
            return Err(e);
        }
        if rep.cancelled() {
            return Err(Cancelled.into());
        }
        set_sparse(&f, false); // every byte is written: an ordinary file again
        Ok(())
    }

    /// One byte range written at its offset; on failure the counted bytes are taken back.
    fn fetch_range(&self, url: &str, f: &File, a: u64, len: u64, bytes: &AtomicU64, rep: &dyn Reporter) -> Result<()> {
        let mut written = 0u64;
        let r = (|| -> Result<()> {
            let resp = self.get(url).header("Range", format!("bytes={}-{}", a, a + len - 1)).call()?;
            let status = resp.status().as_u16();
            if status != 206 {
                bail!("HTTP {status} for a byte range (expected 206)");
            }
            let mut reader = resp.into_body().into_reader();
            let mut buf = vec![0u8; 1 << 20];
            while written < len {
                if rep.cancelled() {
                    return Err(Cancelled.into());
                }
                let k = reader.read(&mut buf)?;
                if k == 0 {
                    break;
                }
                let k = (k as u64).min(len - written) as usize;
                let mut off = 0;
                while off < k {
                    off += f.seek_write(&buf[off..k], a + written + off as u64)?;
                }
                written += k as u64;
                bytes.fetch_add(k as u64, Ordering::Relaxed);
            }
            if written != len {
                bail!("range ended after {written} of {len} bytes");
            }
            Ok(())
        })();
        if r.is_err() {
            bytes.fetch_sub(written, Ordering::Relaxed);
        }
        r
    }
}

/// Marks a file sparse (or back to normal once it has no holes). Best effort: FAT/exFAT have no sparse files.
fn set_sparse(f: &File, on: bool) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::IO::DeviceIoControl;
    use windows_sys::Win32::System::Ioctl::{FILE_SET_SPARSE_BUFFER, FSCTL_SET_SPARSE};
    let b = FILE_SET_SPARSE_BUFFER { SetSparse: on };
    let mut ret = 0u32;
    unsafe {
        DeviceIoControl(
            f.as_raw_handle() as _,
            FSCTL_SET_SPARSE,
            &b as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<FILE_SET_SPARSE_BUFFER>() as u32,
            std::ptr::null_mut(),
            0,
            &mut ret,
            std::ptr::null_mut(),
        );
    }
}

/// A signed CDN URL that stopped working (expired or revoked): resolve the original URL again.
fn expired(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}");
    ["HTTP 401", "HTTP 403", "HTTP 404", "HTTP 410"].iter().any(|c| s.contains(c))
}

fn next_link(h: &str) -> Option<String> {
    h.split(',').find(|p| p.contains("rel=\"next\"")).and_then(|p| {
        let (a, b) = (p.find('<')?, p.find('>')?);
        Some(p[a + 1..b].to_string())
    })
}

pub fn suffixed(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Hashes a file and compares it with the expected digest.
pub fn verify(path: &Path, check: &Check, label: &str, rep: &dyn Reporter) -> Result<()> {
    let got = match check {
        Check::None => return Ok(()),
        Check::Sha256(_) => hash_file::<Sha256>(path, None, label, rep)?,
        Check::GitBlob(_) => {
            let size = path.metadata()?.len();
            hash_file::<Sha1>(path, Some(format!("blob {size}\0").into_bytes()), label, rep)?
        }
    };
    let want = match check {
        Check::Sha256(h) | Check::GitBlob(h) => h.to_ascii_lowercase(),
        Check::None => unreachable!(),
    };
    if got != want {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(suffixed(path, ".done"));
        bail!("{label}: checksum mismatch (got {got}, expected {want}); the file was deleted, run the installer again");
    }
    Ok(())
}

pub fn hash_file<D: Digest>(path: &Path, prefix: Option<Vec<u8>>, label: &str, rep: &dyn Reporter) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let total = f.metadata()?.len();
    let mut h = D::new();
    if let Some(p) = prefix {
        h.update(&p);
    }
    let mut buf = vec![0u8; 8 << 20];
    let mut done = 0u64;
    let what = format!("Verifying {label}");
    loop {
        if rep.cancelled() {
            return Err(Cancelled.into());
        }
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as u64;
        if total > (256 << 20) {
            rep.progress(&what, done, total);
        }
    }
    Ok(hex(&h.finalize()))
}

/// Extracts the zip entries whose names pass `pick` (returning the destination path) and returns what it wrote.
pub fn unzip(zip_path: &Path, mut pick: impl FnMut(&str) -> Option<PathBuf>) -> Result<Vec<PathBuf>> {
    let mut z = zip::ZipArchive::new(File::open(zip_path)?).with_context(|| format!("reading {}", zip_path.display()))?;
    let mut out = Vec::new();
    for i in 0..z.len() {
        let mut e = z.by_index(i)?;
        if e.is_dir() {
            continue;
        }
        let name = e.name()?.replace('\\', "/");
        if name.split('/').any(|c| c == "..") {
            continue;
        }
        if let Some(dest) = pick(&name) {
            if let Some(d) = dest.parent() {
                std::fs::create_dir_all(d)?;
            }
            let tmp = suffixed(&dest, ".new");
            let mut w = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
            std::io::copy(&mut e, &mut w)?;
            drop(w);
            if dest.exists() {
                std::fs::remove_file(&dest).with_context(|| format!("replacing {} (is Inkling running?)", dest.display()))?;
            }
            std::fs::rename(&tmp, &dest)?;
            out.push(dest);
        }
    }
    if out.is_empty() {
        return Err(anyhow!("{}: nothing to extract", zip_path.display()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the time goes: cargo test --release -- --ignored speed_split --nocapture
    #[test]
    #[ignore]
    fn speed_split() {
        struct Quiet;
        impl Reporter for Quiet {
            fn step(&self, _: &str) {}
            fn progress(&self, _: &str, _: u64, _: u64) {}
            fn log(&self, _: &str) {}
        }
        let http = Http::new();
        let url = "https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/4f9088ea66c21f2b99ed657d1941c306b03e84d2/model.safetensors";
        let cdn = http.resolve(url).unwrap();
        // A: one connection, sequential file writes
        let dir = std::env::temp_dir().join(format!("inkling-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = std::time::Instant::now();
        http.try_stream(&cdn, &dir.join("a.bin"), None, "a", &Quiet).unwrap();
        let n = dir.join("a.bin").metadata().unwrap().len() as f64;
        println!("A 1 conn -> file: {:.0} MB/s", n / 1e6 / t.elapsed().as_secs_f64());
        // B: 16 connections x 32 MiB ranges, data discarded
        let t = std::time::Instant::now();
        let total = AtomicU64::new(0);
        std::thread::scope(|s| {
            for i in 0..16u64 {
                let (http, cdn, total) = (&http, &cdn, &total);
                s.spawn(move || {
                    let a = i * CHUNK;
                    let resp = http.get(cdn).header("Range", format!("bytes={}-{}", a, a + CHUNK - 1)).call().unwrap();
                    let mut r = resp.into_body().into_reader();
                    let mut buf = vec![0u8; 1 << 20];
                    loop {
                        let k = r.read(&mut buf).unwrap();
                        if k == 0 {
                            break;
                        }
                        total.fetch_add(k as u64, Ordering::Relaxed);
                    }
                });
            }
        });
        println!("B 16 conn -> memory: {:.0} MB/s", total.load(Ordering::Relaxed) as f64 / 1e6 / t.elapsed().as_secs_f64());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn parses_link_header() {
        let h = r#"<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel="next""#;
        assert_eq!(next_link(h).as_deref(), Some("https://huggingface.co/api/models/x/tree/main?cursor=abc"));
    }

    #[test]
    fn git_blob_sha1_matches_git() {
        // `printf 'hello\n' | git hash-object --stdin`
        let dir = std::env::temp_dir().join(format!("inkling-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("hello.txt");
        std::fs::write(&p, b"hello\n").unwrap();
        struct Quiet;
        impl Reporter for Quiet {
            fn step(&self, _: &str) {}
            fn progress(&self, _: &str, _: u64, _: u64) {}
            fn log(&self, _: &str) {}
        }
        verify(&p, &Check::GitBlob("ce013625030ba8dba906f756967f9e9ca394464a".into()), "hello", &Quiet).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
