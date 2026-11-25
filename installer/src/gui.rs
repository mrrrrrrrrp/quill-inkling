//! The installer window: hardware check, settings, progress, and the uninstall page.

use eframe::egui::{self, Color32, RichText};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::config::{Config, Vision};
use crate::engine::{EngineSource, Toolchain};
use crate::hw::{self, Hardware};
use crate::install;
use crate::plan::{self, Assessment, Support};
use crate::progress::{Meter, Reporter};
use crate::release;
use crate::util::{fmt_bytes, shell_open};

pub enum Start {
    Install(Option<PathBuf>),
    Uninstall(PathBuf),
}

pub fn run(start: Start) -> anyhow::Result<()> {
    let title = match start {
        Start::Install(_) => "Inkling setup",
        Start::Uninstall(_) => "Uninstall Inkling",
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([780.0, 720.0])
            .with_min_inner_size([620.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(title, options, Box::new(move |_cc| Ok(Box::new(App::new(start)))))
        .map_err(|e| anyhow::anyhow!("could not open the window: {e}"))
}

// ---- progress shared with the worker thread ----

#[derive(Default)]
struct Shared {
    steps: Vec<String>,
    current: Option<(String, u64, u64)>,
    log: Vec<String>,
    result: Option<Result<(), String>>,
    meter: Option<Meter>,
}

#[derive(Clone)]
struct GuiReporter {
    shared: Arc<Mutex<Shared>>,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
}

impl Reporter for GuiReporter {
    fn step(&self, name: &str) {
        let mut s = self.shared.lock().unwrap();
        s.steps.push(name.to_string());
        s.current = None;
        s.log.push(format!("== {name}"));
        self.ctx.request_repaint();
    }
    fn progress(&self, label: &str, done: u64, total: u64) {
        self.shared.lock().unwrap().current = Some((label.to_string(), done, total));
        self.ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
    fn log(&self, line: &str) {
        let mut s = self.shared.lock().unwrap();
        s.log.push(line.to_string());
        if s.log.len() > 3000 {
            s.log.drain(..1000);
        }
        self.ctx.request_repaint();
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

// ---- the app ----

#[derive(PartialEq)]
enum Page {
    Form,
    Running,
    Done,
    Uninstall,
    Uninstalling,
}

struct App {
    hw: Hardware,
    assessment: Assessment,
    toolchain: Toolchain,
    page: Page,
    dir: String,
    existing: bool,
    free: Option<u64>,
    cfg: Config,
    desktop: bool,
    start_menu: bool,
    from_source: bool,
    source_dir: String,
    arch: String,
    models_from: String,
    force: bool,
    keep_models: bool,
    shared: Arc<Mutex<Shared>>,
    cancel: Arc<AtomicBool>,
    running_uninstall: bool,
}

impl App {
    fn new(start: Start) -> App {
        let hw = hw::detect();
        let assessment = plan::assess(&hw);
        let (dir, page) = match start {
            Start::Install(d) => (d.unwrap_or_else(install::default_dir), Page::Form),
            Start::Uninstall(d) => (d, Page::Uninstall),
        };
        let mut app = App {
            toolchain: Toolchain::default(),
            arch: assessment.cuda_arch.clone().unwrap_or_else(|| "89".into()),
            from_source: !assessment.prebuilt && assessment.gpu.as_ref().map(|g| g.cc >= (8, 9)).unwrap_or(false),
            hw,
            assessment,
            page,
            dir: String::new(),
            existing: false,
            free: None,
            cfg: Config::default(),
            desktop: true,
            start_menu: true,
            source_dir: crate::engine::find_checkout().map(|p| p.display().to_string()).unwrap_or_default(),
            models_from: String::new(),
            force: false,
            keep_models: false,
            shared: Arc::new(Mutex::new(Shared::default())),
            cancel: Arc::new(AtomicBool::new(false)),
            running_uninstall: false,
        };
        if app.from_source {
            app.toolchain = Toolchain::detect();
        }
        app.set_dir(dir.display().to_string());
        app
    }

    fn set_dir(&mut self, dir: String) {
        self.dir = dir;
        let p = PathBuf::from(&self.dir);
        self.existing = Config::exists(&p);
        self.cfg = crate::base_config(&p, &self.hw, &self.assessment);
        self.free = hw::free_disk_bytes(&p);
    }

    fn start(&mut self, ctx: &egui::Context) {
        *self.shared.lock().unwrap() = Shared::default();
        self.cancel.store(false, Ordering::Relaxed);
        let rep = GuiReporter { shared: self.shared.clone(), cancel: self.cancel.clone(), ctx: ctx.clone() };
        let dir = PathBuf::from(self.dir.trim());
        if self.page == Page::Uninstall {
            let keep = self.keep_models;
            self.page = Page::Uninstalling;
            std::thread::spawn(move || {
                let r = install::uninstall(&dir, keep, &rep);
                rep.shared.lock().unwrap().result = Some(r.map_err(|e| format!("{e:#}")));
                rep.ctx.request_repaint();
            });
            return;
        }
        let engine = if self.from_source {
            EngineSource::Source {
                dir: Some(self.source_dir.trim()).filter(|s| !s.is_empty()).map(PathBuf::from),
                arch: self.arch.trim().to_string(),
            }
        } else {
            EngineSource::Prebuilt { from: None }
        };
        let o = install::Options {
            dir,
            config: self.cfg.clone(),
            engine,
            models_from: Some(self.models_from.trim()).filter(|s| !s.is_empty()).map(PathBuf::from),
            desktop_shortcut: self.desktop,
            start_menu: self.start_menu,
            skip_models: false,
        };
        self.page = Page::Running;
        std::thread::spawn(move || {
            let r = install::run(&o, &rep);
            rep.shared.lock().unwrap().result = Some(r.map_err(|e| format!("{e:#}")));
            rep.ctx.request_repaint();
        });
    }
}

fn support_color(s: Support) -> Color32 {
    match s {
        Support::Confirmed => Color32::from_rgb(40, 160, 80),
        Support::Expected => Color32::from_rgb(60, 130, 220),
        Support::Experimental => Color32::from_rgb(220, 150, 30),
        Support::Unsupported => Color32::from_rgb(220, 60, 60),
    }
}

fn ctx_label(c: u32) -> String {
    match c {
        600_064 => "600k tokens (confirmed)".into(),
        1_048_576 => "1M tokens (the model's maximum)".into(),
        c => format!("{}k tokens", c / 1024),
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // worker finished?
        let finished = self.shared.lock().unwrap().result.is_some();
        if finished && self.page == Page::Running {
            self.page = Page::Done;
        }
        if finished && self.page == Page::Uninstalling {
            self.running_uninstall = true;
        }

        egui::Panel::bottom("buttons").show(ui, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| self.buttons(ui, &ctx));
            ui.add_space(8.0);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(RichText::new("Inkling 2 Mini").strong());
                    ui.label(RichText::new(format!("quill engine · installer {}", release::VERSION)).weak());
                });
                ui.label(
                    "Runs Inkling 2 Mini (abliterated) locally on your NVIDIA GPU: an OpenAI-compatible server with a chat \
                     page in your browser. Confirmed on an RTX 40 with 12 GB + 32 GB RAM: 100+ tok/s, 600k context.",
                );
                ui.add_space(6.0);
                match self.page {
                    Page::Form => self.form(ui),
                    Page::Running | Page::Done | Page::Uninstalling => self.progress(ui),
                    Page::Uninstall => self.uninstall_page(ui),
                }
            });
        });
    }
}

impl App {
    fn hardware(&self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Your computer").strong());
            egui::Grid::new("hw").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                match &self.assessment.gpu {
                    Some(g) => {
                        ui.label("GPU");
                        ui.label(format!("{}  ·  {:.1} GB  ·  {}  ·  driver {}", g.name, g.vram_gib(), g.sm(), g.driver));
                    }
                    None => {
                        ui.label("GPU");
                        ui.label(self.hw.gpu_error.clone().unwrap_or_else(|| "none found".into()));
                    }
                }
                ui.end_row();
                ui.label("Memory");
                ui.label(format!("{:.1} GB", self.hw.ram_gib()));
                ui.end_row();
                ui.label("CPU");
                ui.label(format!(
                    "{}  ·  {} cores{}",
                    self.hw.cpu_name,
                    self.hw.physical_cores,
                    if self.hw.avx2 { "" } else { "  ·  no AVX2" }
                ));
                ui.end_row();
            });
            ui.add_space(4.0);
            let a = &self.assessment;
            ui.label(RichText::new(a.support.label()).strong().color(support_color(a.support)));
            ui.label(&a.summary);
            for n in &a.notes {
                ui.label(format!("•  {n}"));
            }
        });
    }

    fn form(&mut self, ui: &mut egui::Ui) {
        self.hardware(ui);
        ui.add_space(6.0);
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(if self.existing { "Installed: change settings or repair" } else { "Install" }).strong());
            ui.horizontal(|ui| {
                ui.label("Folder");
                let mut d = self.dir.clone();
                if ui.add(egui::TextEdit::singleline(&mut d).desired_width(ui.available_width() - 90.0)).changed() {
                    self.set_dir(d);
                }
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().set_directory(&self.dir).pick_folder() {
                        self.set_dir(p.display().to_string());
                    }
                }
            });
            ui.label(
                RichText::new(format!(
                    "{} free · the models need about {}{}",
                    self.free.map(fmt_bytes).unwrap_or_else(|| "?".into()),
                    fmt_bytes(release::MODELS_BYTES),
                    if self.existing { " (files already there are kept)" } else { "" }
                ))
                .weak(),
            );
            ui.add_space(4.0);
            let fit = self.assessment.max_ctx_fit;
            egui::Grid::new("opts").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
                ui.label("Context window");
                egui::ComboBox::from_id_salt("ctx").selected_text(ctx_label(self.cfg.server.max_ctx)).show_ui(ui, |ui| {
                    for &c in plan::CTX_PRESETS {
                        let ok = c <= fit || fit == 0;
                        let text = if ok { ctx_label(c) } else { format!("{} (too large for this GPU)", ctx_label(c)) };
                        ui.add_enabled_ui(ok, |ui| ui.selectable_value(&mut self.cfg.server.max_ctx, c, text));
                    }
                });
                ui.end_row();
                ui.label("Images");
                egui::ComboBox::from_id_salt("vision").selected_text(self.cfg.server.vision.label()).show_ui(ui, |ui| {
                    for v in [Vision::Cpu, Vision::Gpu, Vision::Off] {
                        ui.selectable_value(&mut self.cfg.server.vision, v, v.label());
                    }
                });
                ui.end_row();
                ui.label("Shortcuts");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.start_menu, "Start menu");
                    ui.checkbox(&mut self.desktop, "Desktop");
                });
                ui.end_row();
            });
            ui.add_space(2.0);
            egui::CollapsingHeader::new("Advanced").show(ui, |ui| self.advanced(ui));
        });
    }

    fn advanced(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.cfg.server;
        egui::Grid::new("adv").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
            ui.label("VRAM cap (GB)");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut s.vram_cap_gib).range(4.0..=96.0).speed(0.05).fixed_decimals(1));
                ui.label(RichText::new("GPU memory of all programs stays under this; card size - 0.7").weak());
            });
            ui.end_row();
            ui.label("CPU threads");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut s.cpu_threads).range(1..=128));
                ui.label(RichText::new("experts that do not fit on the GPU run here; physical cores - 1").weak());
            });
            ui.end_row();
            ui.label("Port");
            ui.add(egui::DragValue::new(&mut s.port).range(1024..=65535));
            ui.end_row();
            ui.label("Engine");
            ui.vertical(|ui| {
                let prebuilt_ok = self.assessment.prebuilt;
                ui.add_enabled_ui(prebuilt_ok, |ui| {
                    ui.radio_value(&mut self.from_source, false, "Prebuilt (RTX 40 and RTX 50)");
                });
                if ui.radio_value(&mut self.from_source, true, "Build from source (Visual Studio 2022 + CUDA 13)").clicked() {
                    self.toolchain = Toolchain::detect();
                }
                if self.from_source {
                    ui.horizontal(|ui| {
                        ui.label("CUDA arch");
                        ui.add(egui::TextEdit::singleline(&mut self.arch).desired_width(90.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Source folder");
                        ui.add(egui::TextEdit::singleline(&mut self.source_dir).hint_text("download from GitHub").desired_width(320.0));
                    });
                    let missing = self.toolchain.missing();
                    if missing.is_empty() {
                        ui.label(RichText::new(format!("Toolchain ready (CUDA {})", self.toolchain.nvcc_version.clone().unwrap_or_default())).color(support_color(Support::Confirmed)));
                    } else {
                        for m in missing {
                            ui.label(RichText::new(format!("Missing: {m}")).color(support_color(Support::Experimental)));
                        }
                    }
                }
            });
            ui.end_row();
            ui.label("Reuse model files");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.models_from).hint_text("a folder that already holds the models").desired_width(320.0));
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.models_from = p.display().to_string();
                    }
                }
            });
            ui.end_row();
        });
        ui.label(RichText::new("More switches live in inkling.toml in the install folder; AGENTS.md explains them.").weak());
    }

    fn progress(&mut self, ui: &mut egui::Ui) {
        let s = self.shared.lock().unwrap();
        let done = s.result.is_some();
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(format!("{} {}", if self.page == Page::Uninstalling { "Removing from" } else { "Installing to" }, self.dir)).strong());
            for (i, st) in s.steps.iter().enumerate() {
                let last = i + 1 == s.steps.len();
                let (mark, color) = match (&s.result, last) {
                    (Some(Err(_)), true) => ("✖", support_color(Support::Unsupported)),
                    (None, true) => ("▶", ui.visuals().strong_text_color()),
                    _ => ("✔", support_color(Support::Confirmed)),
                };
                ui.label(RichText::new(format!("{mark}  {st}")).color(color));
            }
        });
        drop(s);
        let mut s = self.shared.lock().unwrap();
        if let (Some((label, d, t)), None) = (s.current.clone(), &s.result) {
            let text = Meter::text(&label, d, t, &mut s.meter);
            ui.add_space(4.0);
            ui.label(&label);
            let frac = if t > 0 { d as f32 / t as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(text).animate(t == 0));
        }
        match &s.result {
            Some(Ok(())) if self.page == Page::Uninstalling => {
                ui.add_space(8.0);
                ui.label(RichText::new("Inkling was removed.").strong().color(support_color(Support::Confirmed)));
            }
            Some(Ok(())) => {
                ui.add_space(8.0);
                ui.label(RichText::new("Inkling is installed.").heading().color(support_color(Support::Confirmed)));
                ui.label(format!(
                    "Start it from the Start menu or desktop (\"Inkling\"). The server window shows the log; the first load takes \
                     about a minute, then the chat opens at http://127.0.0.1:{0}/.\nOpenAI-compatible API: http://127.0.0.1:{0}/v1, \
                     model id inkling-2-mini-abliterated.",
                    self.cfg.server.port
                ));
            }
            Some(Err(e)) => {
                ui.add_space(8.0);
                ui.label(RichText::new("Something went wrong").strong().color(support_color(Support::Unsupported)));
                ui.label(e);
                if !self.running_uninstall {
                    ui.label(RichText::new("Downloads resume where they stopped when you try again.").weak());
                }
            }
            None => {}
        }
        ui.add_space(6.0);
        egui::CollapsingHeader::new("Log").default_open(!done || matches!(s.result, Some(Err(_)))).show(ui, |ui| {
            egui::ScrollArea::vertical().max_height(220.0).stick_to_bottom(true).auto_shrink([false, true]).show(ui, |ui| {
                for l in &s.log {
                    ui.label(RichText::new(l).monospace().size(11.5));
                }
            });
        });
    }

    fn uninstall_page(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Uninstall").strong());
            ui.label(format!("Removes Inkling from {}: the engine, the chat page, the settings, the shortcuts and the Apps & features entry. Other files in the folder are left alone.", self.dir));
            ui.checkbox(&mut self.keep_models, format!("Keep the downloaded models (about {}) for a later install", fmt_bytes(release::MODELS_BYTES)));
        });
    }

    fn buttons(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let dir = PathBuf::from(self.dir.trim());
        let result = self.shared.lock().unwrap().result.clone();
        match self.page {
            Page::Form => {
                if self.existing && ui.button("Uninstall…").clicked() {
                    self.page = Page::Uninstall;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let unsupported = self.assessment.support == Support::Unsupported;
                    let ok = (!unsupported || self.force) && !self.dir.trim().is_empty();
                    let label = if self.existing { "Update / repair" } else { "Install" };
                    if ui.add_enabled(ok, egui::Button::new(RichText::new(label).strong()).min_size([120.0, 30.0].into())).clicked() {
                        self.start(ctx);
                    }
                    if unsupported {
                        ui.checkbox(&mut self.force, "Install anyway");
                    }
                });
            }
            Page::Running | Page::Uninstalling if result.is_none() => {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let cancelling = self.cancel.load(Ordering::Relaxed);
                    if ui.add_enabled(!cancelling && self.page == Page::Running, egui::Button::new(if cancelling { "Stopping…" } else { "Cancel" })).clicked() {
                        self.cancel.store(true, Ordering::Relaxed);
                    }
                });
            }
            Page::Done => {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match result {
                    Some(Ok(())) => {
                        if ui.add(egui::Button::new(RichText::new("Launch Inkling").strong()).min_size([140.0, 30.0].into())).clicked() {
                            let _ = std::process::Command::new(dir.join("Inkling.exe")).current_dir(&dir).spawn();
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Open folder").clicked() {
                            shell_open(&dir.display().to_string());
                        }
                        if ui.button("Close").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                    _ => {
                        if ui.button("Back").clicked() {
                            self.page = Page::Form;
                            self.set_dir(self.dir.clone());
                        }
                    }
                });
            }
            Page::Uninstall => {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new(RichText::new("Uninstall").strong()).min_size([120.0, 30.0].into())).clicked() {
                        self.start(ctx);
                    }
                    if ui.button("Cancel").clicked() {
                        if Config::exists(&dir) && install::default_dir() == dir {
                            self.page = Page::Form;
                        } else {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                });
            }
            Page::Running | Page::Uninstalling => {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            }
        }
    }
}
