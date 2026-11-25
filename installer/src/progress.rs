//! How the install steps report to the window or the console.

use std::io::{IsTerminal, Write};
use std::sync::Mutex;
use std::time::Instant;

use crate::util::{fmt_bytes, fmt_duration};

pub trait Reporter: Send + Sync {
    /// a new install step begins (the previous one is done)
    fn step(&self, name: &str);
    /// bytes of the current transfer; total 0 = unknown
    fn progress(&self, label: &str, done: u64, total: u64);
    fn log(&self, line: &str);
    fn cancelled(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct Cancelled;
impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled; run the installer again to resume where it stopped")
    }
}
impl std::error::Error for Cancelled {}

/// Rate and time left of one transfer, from when it (re)started.
pub struct Meter {
    label: String,
    start: Instant,
    start_done: u64,
}

impl Meter {
    pub fn text(label: &str, done: u64, total: u64, meter: &mut Option<Meter>) -> String {
        let m = match meter {
            Some(m) if m.label == label && done >= m.start_done => m,
            _ => meter.insert(Meter { label: label.to_string(), start: Instant::now(), start_done: done }),
        };
        let secs = m.start.elapsed().as_secs_f64();
        let rate = if secs > 1.0 { (done - m.start_done) as f64 / secs } else { 0.0 };
        let mut s = if total > 0 {
            format!("{} / {}", fmt_bytes(done), fmt_bytes(total))
        } else {
            fmt_bytes(done)
        };
        if rate > 0.0 {
            s += &format!(" · {}/s", fmt_bytes(rate as u64));
            if total > done {
                s += &format!(" · {} left", fmt_duration((total - done) as f64 / rate));
            }
        }
        s
    }
}

/// Console reporter: a live line on a terminal, a line every 10% when the output is captured.
pub struct ConsoleReporter {
    state: Mutex<(Option<Meter>, u64, Instant)>,
    tty: bool,
}

impl ConsoleReporter {
    pub fn new() -> Self {
        ConsoleReporter { state: Mutex::new((None, u64::MAX, Instant::now())), tty: std::io::stdout().is_terminal() }
    }
}

impl Reporter for ConsoleReporter {
    fn step(&self, name: &str) {
        let mut st = self.state.lock().unwrap();
        st.1 = u64::MAX;
        if self.tty {
            print!("\r\x1b[2K");
        }
        println!("== {name}");
    }
    fn progress(&self, label: &str, done: u64, total: u64) {
        let mut st = self.state.lock().unwrap();
        let (meter, last_decile, last_print) = &mut *st;
        let text = Meter::text(label, done, total, meter);
        if self.tty {
            if last_print.elapsed().as_millis() >= 200 || done == total {
                print!("\r\x1b[2K   {label}: {text}");
                let _ = std::io::stdout().flush();
                *last_print = Instant::now();
            }
        } else {
            let decile = if total > 0 { done * 10 / total } else { done >> 30 };
            if decile != *last_decile {
                println!("   {label}: {text}");
                *last_decile = decile;
            }
        }
    }
    fn log(&self, line: &str) {
        if self.tty {
            print!("\r\x1b[2K");
        }
        println!("   {line}");
    }
}
