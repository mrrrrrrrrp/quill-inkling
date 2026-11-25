use std::os::windows::process::CommandExt;
use std::process::Command;

pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Runs a console program without flashing a console window (this exe is a windowed program).
pub fn hidden(cmd: &mut Command) -> &mut Command {
    cmd.creation_flags(CREATE_NO_WINDOW)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= GIB {
        format!("{:.1} GB", b / GIB)
    } else if b >= 1024.0 * 1024.0 {
        format!("{:.0} MB", b / (1024.0 * 1024.0))
    } else if b >= 1024.0 {
        format!("{:.0} KB", b / 1024.0)
    } else {
        format!("{b:.0} B")
    }
}

pub fn fmt_duration(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{} min {:02} s", s / 60, s % 60)
    } else {
        format!("{s} s")
    }
}

/// Opens a URL or folder with its default program.
pub fn shell_open(target: &str) {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
    let w = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let (op, t) = (w("open"), w(target));
    unsafe { ShellExecuteW(std::ptr::null_mut(), op.as_ptr(), t.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL) };
}
