//! Windows integration: consoles, the job object that ties the server to the launcher, shortcuts, the
//! "Apps & features" entry.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

use crate::release;
use crate::util::hidden;

const UNINSTALL_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Inkling-quill";

// ---- consoles (this exe is a windowed program, so it has none by default) ----

static STARTED_WITH_CONSOLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Call first thing: a console at start means this is the console build (inkling-cli.exe) run from a terminal.
pub fn note_start_console() {
    use windows_sys::Win32::System::Console::GetConsoleWindow;
    let has = unsafe { !GetConsoleWindow().is_null() };
    STARTED_WITH_CONSOLE.store(has, std::sync::atomic::Ordering::Relaxed);
}

pub fn started_with_console() -> bool {
    STARTED_WITH_CONSOLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Writes a copy of an exe with the console subsystem: PowerShell and cmd wait for it and capture its output
/// (they do neither for windowed programs). Only the PE header's Subsystem field differs.
pub fn write_console_copy(src: &Path, dest: &Path) -> Result<()> {
    let mut b = std::fs::read(src).with_context(|| format!("reading {}", src.display()))?;
    let pe = u32::from_le_bytes(b.get(0x3c..0x40).context("not an exe")?.try_into()?) as usize;
    if b.get(pe..pe + 4) != Some(b"PE\0\0".as_slice()) {
        bail!("{} is not a PE file", src.display());
    }
    let subsystem = pe + 24 + 68; // optional header + 68: the same offset in PE32 and PE32+
    if b.len() < subsystem + 2 {
        bail!("{}: truncated PE header", src.display());
    }
    b[subsystem..subsystem + 2].copy_from_slice(&3u16.to_le_bytes()); // IMAGE_SUBSYSTEM_WINDOWS_CUI
    if dest.exists() {
        std::fs::remove_file(dest).with_context(|| format!("replacing {}", dest.display()))?;
    }
    std::fs::write(dest, b).with_context(|| format!("writing {}", dest.display()))
}

/// For command-line use: keep redirected output (pipes, files), else write to the calling terminal.
pub fn attach_parent_console() {
    use windows_sys::Win32::Storage::FileSystem::{GetFileType, FILE_TYPE_DISK, FILE_TYPE_PIPE};
    use windows_sys::Win32::System::Console::{AttachConsole, GetStdHandle, ATTACH_PARENT_PROCESS, STD_OUTPUT_HANDLE};
    unsafe {
        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        if !h.is_null() && h as isize != -1 {
            let t = GetFileType(h);
            if t == FILE_TYPE_PIPE || t == FILE_TYPE_DISK {
                return;
            }
        }
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

/// A console window of our own (the launcher shows the server's log in it).
pub fn new_console(title: &str) {
    use windows_sys::Win32::System::Console::{AllocConsole, FreeConsole, SetConsoleTitleW};
    unsafe {
        FreeConsole();
        AllocConsole();
        let w: Vec<u16> = title.encode_utf16().chain(Some(0)).collect();
        SetConsoleTitleW(w.as_ptr());
    }
}

// ---- job object: the server dies with the launcher (window closed, crash, Task Manager) ----

/// Held for its lifetime: the handle closes when the launcher exits, which kills the job's processes.
pub struct Job(#[allow(dead_code)] windows_sys::Win32::Foundation::HANDLE);

impl Job {
    pub fn kill_on_close(child: &std::process::Child) -> Option<Job> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0 {
                return None;
            }
            Some(Job(job))
        }
    }
}

// ---- shortcuts (WScript.Shell through PowerShell; paths passed in environment variables) ----

pub fn create_shortcuts(dir: &Path, desktop: bool, start_menu: bool) -> Result<()> {
    if !desktop && !start_menu {
        return Ok(());
    }
    let script = r#"
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject WScript.Shell
function L($path, $arguments, $desc) {
  $s = $w.CreateShortcut($path); $s.TargetPath = $env:INK_EXE; $s.Arguments = $arguments
  $s.WorkingDirectory = $env:INK_DIR; $s.Description = $desc; $s.IconLocation = "$($env:INK_EXE),0"; $s.Save()
}
if ($env:INK_SM -eq '1') {
  $sm = Join-Path ([Environment]::GetFolderPath('Programs')) 'Inkling'
  New-Item -ItemType Directory -Force $sm | Out-Null
  L (Join-Path $sm 'Inkling.lnk') '' 'Chat with Inkling 2 Mini (starts the local server)'
  L (Join-Path $sm 'Inkling Settings.lnk') 'setup' 'Change settings, repair or uninstall Inkling'
}
if ($env:INK_DESK -eq '1') {
  L (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Inkling.lnk') '' 'Chat with Inkling 2 Mini (starts the local server)'
}
"#;
    powershell(script, dir, &[("INK_SM", if start_menu { "1" } else { "0" }), ("INK_DESK", if desktop { "1" } else { "0" })])
}

pub fn remove_shortcuts(dir: &Path) -> Result<()> {
    let script = r#"
$w = New-Object -ComObject WScript.Shell
$sm = Join-Path ([Environment]::GetFolderPath('Programs')) 'Inkling'
if (Test-Path $sm) { Remove-Item -Recurse -Force $sm }
$d = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Inkling.lnk'
if ((Test-Path $d) -and ($w.CreateShortcut($d).TargetPath -eq $env:INK_EXE)) { Remove-Item -Force $d }
"#;
    powershell(script, dir, &[])
}

fn powershell(script: &str, dir: &Path, env: &[(&str, &str)]) -> Result<()> {
    let exe = dir.join("Inkling.exe");
    let out = hidden(
        Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script])
            .env("INK_EXE", &exe)
            .env("INK_DIR", dir)
            .envs(env.iter().copied()),
    )
    .output()
    .context("running PowerShell")?;
    if !out.status.success() {
        bail!("PowerShell: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

// ---- Apps & features ----

pub fn register_uninstall(dir: &Path, size_bytes: u64) -> Result<()> {
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};
    let (k, _) = RegKey::predef(HKEY_CURRENT_USER).create_subkey(UNINSTALL_KEY)?;
    let exe = dir.join("Inkling.exe");
    let q = format!("\"{}\"", exe.display());
    k.set_value("DisplayName", &"Inkling 2 Mini (quill)")?;
    k.set_value("DisplayVersion", &release::VERSION)?;
    k.set_value("Publisher", &"quill-inkling contributors")?;
    k.set_value("InstallLocation", &dir.display().to_string())?;
    k.set_value("DisplayIcon", &exe.display().to_string())?;
    k.set_value("UninstallString", &format!("{q} uninstall"))?;
    k.set_value("ModifyPath", &format!("{q} setup"))?;
    k.set_value("NoRepair", &1u32)?;
    k.set_value("EstimatedSize", &((size_bytes / 1024).min(u32::MAX as u64) as u32))?;
    if release::repo_known() {
        k.set_value("URLInfoAbout", &format!("https://github.com/{}", release::REPO))?;
    }
    Ok(())
}

pub fn unregister_uninstall() {
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};
    let _ = RegKey::predef(HKEY_CURRENT_USER).delete_subkey_all(UNINSTALL_KEY);
}

/// The install folder from the "Apps & features" entry, if Inkling is installed.
pub fn installed_dir() -> Option<std::path::PathBuf> {
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};
    let k = RegKey::predef(HKEY_CURRENT_USER).open_subkey(UNINSTALL_KEY).ok()?;
    let d: String = k.get_value("InstallLocation").ok()?;
    Some(d.into())
}

/// Deletes this exe (and then its folder, if empty) once this process has exited, however long the window stays open.
pub fn delete_self_later(exe: &Path, dir: &Path) {
    let script = "Wait-Process -Id $env:INK_PID -ErrorAction SilentlyContinue; Start-Sleep -Milliseconds 500; \
                  Remove-Item -LiteralPath $env:INK_EXE -Force -ErrorAction SilentlyContinue; \
                  try { [IO.Directory]::Delete($env:INK_DIR) } catch {}";
    let _ = hidden(
        Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script])
            .env("INK_PID", std::process::id().to_string())
            .env("INK_EXE", exe)
            .env("INK_DIR", dir),
    )
    .spawn();
}
