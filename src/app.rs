//! The egui application: controls, 2D/3D previews, background building and export.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use eframe::egui::{
    self, Color32, Key, KeyboardShortcut, Modifiers, Pos2, Rect, Sense, Stroke, Vec2,
};
use eframe::{egui_glow, glow};
use glam::{DVec2, Vec3};

use crate::imaging::{Adjustments, GrayImage, MAX_IMAGE_DIM};
use crate::lithophane::{Fit, ImageMap, LithoParams};
use crate::mesh::{earcut_raw, group_loops, triangulate};
use crate::model::{self, BuildInput, Settings, ShapeKind, ShapeSettings};
use crate::project;
use crate::shape::{AxisChoice, Shape};
use crate::supports::SupportParams;
use crate::viewer::{self, Camera, Scene, SharedScene, Uniforms};

/// Rough number of heightmap cells used for the interactive preview.
const PREVIEW_CELLS: f64 = 150_000.0;

const SHORTCUT_NEW: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::N);
const SHORTCUT_OPEN: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
const SHORTCUT_SAVE: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
const SHORTCUT_SAVE_AS: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::S);

const KEY_PROJECT: &str = "project_path";
const KEY_RECENT: &str = "recent_projects";
const MAX_RECENT: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Framing,
    Backlit,
    Model,
}

enum Command {
    New,
    Open,
    OpenPath(PathBuf),
    Save,
    SaveAs,
}

/// Replaces the current project; waits for confirmation while there are unsaved changes.
enum ProjectAction {
    New,
    Open(PathBuf),
}

#[derive(Clone, PartialEq)]
struct BuildKey {
    shape_gen: u64,
    image_gen: u64,
    adjustments: Adjustments,
    litho: LithoParams,
    supports: SupportParams,
    res: f64,
}

struct PreviewResult {
    key: BuildKey,
    vertices: Vec<f32>,
    bounds: Option<(Vec3, Vec3)>,
    backlit: (usize, usize, Vec<u8>),
    triangles: usize,
    millis: u128,
}

struct PreviewInfo {
    res: f64,
    triangles: usize,
    millis: u128,
}

pub struct App {
    s: Settings,
    project_path: Option<PathBuf>,
    /// Settings as stored in `project_path` (defaults for an unsaved project).
    saved: Settings,
    recent: Vec<PathBuf>,
    pending: Option<ProjectAction>,
    window_title: String,
    original_image: Option<Arc<GrayImage>>,
    image: Arc<GrayImage>,
    display_image: Arc<GrayImage>,
    image_gen: u64,
    image_tex: Option<egui::TextureHandle>,
    image_tex_key: Option<(u64, Adjustments)>,
    histogram: Vec<u32>,

    stl: Option<Arc<Vec<[[f32; 3]; 3]>>>,
    shape: Arc<Shape>,
    shape_key: Option<(ShapeSettings, bool)>,
    shape_gen: u64,
    shape_error: Option<String>,
    detected_axis: Option<usize>,
    overlay: Option<(u64, Vec<DVec2>, Vec<[u32; 3]>)>,

    job: Option<(BuildKey, mpsc::Receiver<PreviewResult>)>,
    built_key: Option<BuildKey>,
    preview: Option<PreviewInfo>,
    backlit_tex: Option<egui::TextureHandle>,
    scene: SharedScene,
    camera: Camera,
    fit_camera: bool,
    last_bounds: Option<(Vec3, Vec3)>,
    backlit_3d: bool,
    tab: Tab,

    export_job: Option<mpsc::Receiver<Result<String, String>>>,
    status: Option<(String, bool)>,
}

impl App {
    /// `open` is a project file passed on the command line.
    pub fn new(cc: &eframe::CreationContext<'_>, open: Option<PathBuf>) -> App {
        // The working state of the last session is restored, saved to a project or not.
        let s: Settings = cc
            .storage
            .and_then(|st| eframe::get_value(st, eframe::APP_KEY))
            .unwrap_or_default();
        let project_path: Option<PathBuf> = cc
            .storage
            .and_then(|st| eframe::get_value(st, KEY_PROJECT))
            .flatten();
        let recent: Vec<PathBuf> = cc
            .storage
            .and_then(|st| eframe::get_value(st, KEY_RECENT))
            .unwrap_or_default();
        // Unsaved changes are measured against the project file as it is on disk.
        let saved = project_path.as_deref().and_then(|p| project::load(p).ok());
        let pattern = Arc::new(model::test_pattern());
        let mut app = App {
            s,
            project_path: saved.as_ref().and(project_path),
            saved: saved.unwrap_or_default(),
            recent,
            pending: None,
            window_title: String::new(),
            original_image: None,
            image: pattern.clone(),
            display_image: pattern,
            image_gen: 0,
            image_tex: None,
            image_tex_key: None,
            histogram: Vec::new(),
            stl: None,
            shape: Arc::new(Shape::rectangle(108.0, 144.0, 0.0)),
            shape_key: None,
            shape_gen: 0,
            shape_error: None,
            detected_axis: None,
            overlay: None,
            job: None,
            built_key: None,
            preview: None,
            backlit_tex: None,
            scene: Arc::new(Mutex::new(Scene::default())),
            camera: Camera::default(),
            fit_camera: true,
            last_bounds: None,
            backlit_3d: false,
            tab: Tab::Model,
            export_job: None,
            status: None,
        };
        let problems = app.load_referenced_files();
        if !problems.is_empty() {
            app.set_status(problems.join("; "), true);
        }
        if let Some(p) = open {
            app.request(ProjectAction::Open(p));
        }
        app
    }

    fn set_status(&mut self, msg: impl Into<String>, error: bool) {
        self.status = Some((msg.into(), error));
    }

    fn load_image(&mut self, path: &Path) -> Result<(), String> {
        let img = GrayImage::load(path)?;
        self.original_image = Some(Arc::new(img.downscaled(MAX_IMAGE_DIM)));
        self.s.image_path = Some(path.to_path_buf());
        self.apply_image_orientation();
        Ok(())
    }

    fn open_image(&mut self, path: &Path) {
        match self.load_image(path) {
            Ok(()) => self.set_status(format!("Loaded image {}", file_name(path)), false),
            Err(e) => self.set_status(e, true),
        }
    }

    fn apply_image_orientation(&mut self) {
        let Some(orig) = &self.original_image else {
            return;
        };
        let mut img = (**orig).clone();
        for _ in 0..self.s.image_quarter_turns % 4 {
            img = img.rotated_cw();
        }
        self.display_image = Arc::new(img.downscaled(1024));
        self.image = Arc::new(img);
        self.image_gen += 1;
    }

    fn load_stl(&mut self, path: &Path) -> Result<(), String> {
        let tris = crate::mesh::read_stl(path)?;
        self.stl = Some(Arc::new(tris));
        self.s.shape.stl_path = Some(path.to_path_buf());
        self.shape_key = None;
        Ok(())
    }

    fn open_stl(&mut self, path: &Path) {
        match self.load_stl(path) {
            Ok(()) => {
                self.s.shape.kind = ShapeKind::Stl;
                self.set_status(format!("Loaded outline {}", file_name(path)), false);
            }
            Err(e) => self.set_status(e, true),
        }
    }

    /// Loads the picture and outline named in the settings; returns what went wrong.
    fn load_referenced_files(&mut self) -> Vec<String> {
        let mut problems = Vec::new();
        if let Some(p) = self.s.image_path.clone()
            && let Err(e) = self.load_image(&p)
        {
            problems.push(e);
            self.s.image_path = None;
        }
        if let Some(p) = self.s.shape.stl_path.clone()
            && let Err(e) = self.load_stl(&p)
        {
            problems.push(e);
            self.s.shape.stl_path = None;
        }
        if self.stl.is_none() && self.s.shape.kind == ShapeKind::Stl {
            self.s.shape.kind = ShapeKind::Rectangle;
        }
        problems
    }

    // ------------------------------------------------------------- projects

    fn is_dirty(&self) -> bool {
        self.s != self.saved
    }

    fn project_name(&self) -> String {
        self.project_path
            .as_deref()
            .map(file_name)
            .unwrap_or_else(|| "Untitled".into())
    }

    fn run(&mut self, cmd: Command) {
        match cmd {
            Command::New => self.request(ProjectAction::New),
            Command::Open => {
                if let Some(p) = self.project_dialog("Open project").pick_file() {
                    self.request(ProjectAction::Open(p));
                }
            }
            Command::OpenPath(p) => self.request(ProjectAction::Open(p)),
            Command::Save => {
                self.save_project();
            }
            Command::SaveAs => {
                self.save_project_as();
            }
        }
    }

    fn request(&mut self, action: ProjectAction) {
        if self.is_dirty() {
            self.pending = Some(action);
        } else {
            self.perform(action);
        }
    }

    fn perform(&mut self, action: ProjectAction) {
        match action {
            ProjectAction::New => {
                self.replace_settings(Settings::default());
                self.project_path = None;
                self.saved = self.s.clone();
                self.set_status("New project", false);
            }
            ProjectAction::Open(p) => self.open_project(&p),
        }
    }

    /// Starts over from `s`, dropping the loaded picture and outline.
    fn replace_settings(&mut self, s: Settings) {
        self.s = s;
        self.original_image = None;
        let pattern = Arc::new(model::test_pattern());
        self.image = pattern.clone();
        self.display_image = pattern;
        self.image_gen += 1;
        self.stl = None;
        self.shape_key = None;
        self.detected_axis = None;
        self.fit_camera = true;
    }

    fn open_project(&mut self, path: &Path) {
        let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        match project::load(&path) {
            Ok(s) => {
                self.saved = s.clone();
                self.replace_settings(s);
                let problems = self.load_referenced_files();
                self.project_path = Some(path.clone());
                self.add_recent(&path);
                if problems.is_empty() {
                    self.set_status(format!("Opened project {}", file_name(&path)), false);
                } else {
                    self.set_status(
                        format!(
                            "Opened project {}, but: {}",
                            file_name(&path),
                            problems.join("; ")
                        ),
                        true,
                    );
                }
            }
            Err(e) => {
                if !path.exists() {
                    self.recent.retain(|p| p != &path);
                }
                self.set_status(e, true);
            }
        }
    }

    /// Returns false if the project was not saved (cancelled or failed).
    fn save_project(&mut self) -> bool {
        match self.project_path.clone() {
            Some(p) => self.write_project(&p),
            None => self.save_project_as(),
        }
    }

    fn save_project_as(&mut self) -> bool {
        let stem = self
            .project_path
            .as_ref()
            .or(self.s.image_path.as_ref())
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("lithophane")
            .to_owned();
        let dialog = self
            .project_dialog("Save project")
            .set_file_name(format!("{stem}.{}", project::EXTENSION));
        let Some(mut path) = dialog.save_file() else {
            return false;
        };
        if path.extension().is_none() {
            path.set_extension(project::EXTENSION);
        }
        self.write_project(&path)
    }

    fn write_project(&mut self, path: &Path) -> bool {
        match project::save(path, &self.s) {
            Ok(()) => {
                let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
                self.saved = self.s.clone();
                self.add_recent(&path);
                self.set_status(format!("Saved project {}", file_name(&path)), false);
                self.project_path = Some(path);
                true
            }
            Err(e) => {
                self.set_status(e, true);
                false
            }
        }
    }

    fn project_dialog(&self, title: &str) -> rfd::FileDialog {
        let mut dialog = rfd::FileDialog::new()
            .set_title(title)
            .add_filter("Lithophane project", &[project::EXTENSION]);
        let dir = self
            .project_path
            .as_ref()
            .or(self.recent.first())
            .or(self.s.image_path.as_ref())
            .and_then(|p| p.parent());
        if let Some(dir) = dir {
            dialog = dialog.set_directory(dir);
        }
        dialog
    }

    fn add_recent(&mut self, path: &Path) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(MAX_RECENT);
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if self.pending.is_some() {
            return;
        }
        // Most specific first: Ctrl+S would also match Ctrl+Shift+S.
        let cmd = ctx.input_mut(|i| {
            if i.consume_shortcut(&SHORTCUT_SAVE_AS) {
                Some(Command::SaveAs)
            } else if i.consume_shortcut(&SHORTCUT_SAVE) {
                Some(Command::Save)
            } else if i.consume_shortcut(&SHORTCUT_OPEN) {
                Some(Command::Open)
            } else if i.consume_shortcut(&SHORTCUT_NEW) {
                Some(Command::New)
            } else {
                None
            }
        });
        if let Some(cmd) = cmd {
            self.run(cmd);
        }
    }

    fn unsaved_changes_dialog(&mut self, ctx: &egui::Context) {
        let Some(action) = &self.pending else {
            return;
        };
        let question = format!(
            "Save the changes to {} before {}?",
            self.project_name(),
            match action {
                ProjectAction::New => "starting a new project".to_owned(),
                ProjectAction::Open(p) => format!("opening {}", file_name(p)),
            }
        );
        #[derive(PartialEq)]
        enum Choice {
            Save,
            Discard,
            Cancel,
        }
        let mut choice = None;
        let modal = egui::Modal::new(egui::Id::unique("unsaved_changes")).show(ctx, |ui| {
            ui.set_max_width(380.0);
            ui.heading("Unsaved changes");
            ui.add_space(4.0);
            ui.label(question);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    choice = Some(Choice::Save);
                }
                if ui.button("Don't save").clicked() {
                    choice = Some(Choice::Discard);
                }
                if ui.button("Cancel").clicked() {
                    choice = Some(Choice::Cancel);
                }
            });
        });
        if choice.is_none() && modal.should_close() {
            choice = Some(Choice::Cancel);
        }
        let proceed = match choice {
            None => return,
            Some(Choice::Cancel) => false,
            Some(Choice::Discard) => true,
            // Stay in the dialog if saving was cancelled or failed.
            Some(Choice::Save) => {
                if !self.save_project() {
                    return;
                }
                true
            }
        };
        if let Some(action) = self.pending.take()
            && proceed
        {
            self.perform(action);
        }
    }

    fn open_image_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new().set_title("Open image").add_filter(
            "Images",
            &[
                "png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp", "tga",
            ],
        );
        if let Some(dir) = self.s.image_path.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        if let Some(p) = dialog.pick_file() {
            self.open_image(&p);
        }
    }

    fn open_stl_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Open outline STL (extruded polygon)")
            .add_filter("STL", &["stl", "STL"]);
        if let Some(dir) = self.s.shape.stl_path.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        if let Some(p) = dialog.pick_file() {
            self.open_stl(&p);
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for f in dropped {
            let path = f.path().to_path_buf();
            if path.as_os_str().is_empty() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if project::is_project_file(&path) {
                self.request(ProjectAction::Open(path));
            } else if ext == "stl" {
                self.open_stl(&path);
            } else {
                self.open_image(&path);
            }
        }
    }

    fn update_shape(&mut self) {
        let key = (self.s.shape.clone(), self.stl.is_some());
        if self.shape_key.as_ref() == Some(&key) {
            return;
        }
        self.shape_key = Some(key);
        match model::make_shape(&self.s.shape, self.stl.as_ref().map(|t| t.as_slice())) {
            Ok((shape, axis)) => {
                self.shape = Arc::new(shape);
                self.detected_axis = axis;
                self.shape_error = None;
                self.shape_gen += 1;
                self.fit_camera = true;
            }
            Err(e) => self.shape_error = Some(e),
        }
    }

    fn update_image_texture(&mut self, ctx: &egui::Context) {
        let key = (self.image_gen, self.s.adjustments);
        if self.image_tex_key == Some(key) {
            return;
        }
        let adjusted = self.display_image.adjusted(&self.s.adjustments);
        self.histogram = adjusted.histogram(64);
        let img =
            egui::ColorImage::from_rgba_unmultiplied([adjusted.w, adjusted.h], &adjusted.to_rgba());
        self.image_tex = Some(ctx.load_texture("image", img, egui::TextureOptions::LINEAR));
        self.image_tex_key = Some(key);
    }

    fn preview_res(&self) -> f64 {
        let size = self.shape.size();
        (size.x * size.y / PREVIEW_CELLS)
            .sqrt()
            .max(self.s.resolution)
    }

    fn build_input(&self) -> BuildInput {
        BuildInput {
            shape: self.shape.clone(),
            image: self.image.clone(),
            adjustments: self.s.adjustments,
            litho: self.s.litho.clone(),
            supports: self.s.supports.clone(),
        }
    }

    fn current_key(&self) -> BuildKey {
        BuildKey {
            shape_gen: self.shape_gen,
            image_gen: self.image_gen,
            adjustments: self.s.adjustments,
            litho: self.s.litho.clone(),
            supports: self.s.supports.clone(),
            res: self.preview_res(),
        }
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        if let Some((key, rx)) = &self.job {
            match rx.try_recv() {
                Ok(r) => {
                    self.scene.lock().unwrap().set_vertices(r.vertices);
                    self.last_bounds = r.bounds;
                    if self.fit_camera {
                        self.reset_camera();
                    }
                    let (w, h, rgba) = r.backlit;
                    let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
                    self.backlit_tex =
                        Some(ctx.load_texture("backlit", img, egui::TextureOptions::LINEAR));
                    self.preview = Some(PreviewInfo {
                        res: r.key.res,
                        triangles: r.triangles,
                        millis: r.millis,
                    });
                    self.built_key = Some(r.key);
                    self.job = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.built_key = Some(key.clone());
                    self.job = None;
                    self.set_status("Building the preview failed (see terminal output)", true);
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(rx) = &self.export_job {
            match rx.try_recv() {
                Ok(Ok(msg)) => {
                    self.export_job = None;
                    self.set_status(msg, false);
                }
                Ok(Err(msg)) => {
                    self.export_job = None;
                    self.set_status(msg, true);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.export_job = None;
                    self.set_status("Export failed (see terminal output)", true);
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    fn reset_camera(&mut self) {
        if let Some((lo, hi)) = self.last_bounds {
            self.camera = Camera::default();
            self.camera.fit(lo, hi);
            self.fit_camera = false;
        }
    }

    fn start_preview_if_needed(&mut self, ctx: &egui::Context) {
        if self.job.is_some() || self.shape.is_empty() {
            return;
        }
        let key = self.current_key();
        if self.built_key.as_ref() == Some(&key) {
            return;
        }
        let input = self.build_input();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let job_key = key.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let m = model::build_model(&input, job_key.res);
            let mut meshes = vec![(&m.lithophane, viewer::KIND_LITHOPHANE)];
            meshes.extend(m.supports.iter().map(|s| (s, viewer::KIND_SUPPORT)));
            let vertices = viewer::vertex_data(&meshes);
            let all: Vec<_> = meshes.iter().map(|(m, _)| *m).collect();
            let result = PreviewResult {
                bounds: viewer::mesh_bounds(&all),
                backlit: m.heightmap.backlit_rgba(),
                triangles: m.triangle_count(),
                millis: t0.elapsed().as_millis(),
                vertices,
                key: job_key,
            };
            tx.send(result).ok();
            ctx.request_repaint();
        });
        self.job = Some((key, rx));
    }

    fn start_export(&mut self) {
        if self.export_job.is_some() {
            return;
        }
        let default_name = self
            .s
            .image_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}_lithophane.stl"))
            .unwrap_or_else(|| "lithophane.stl".into());
        let mut dialog = rfd::FileDialog::new()
            .set_title("Export lithophane STL")
            .add_filter("STL", &["stl"])
            .set_file_name(default_name);
        if let Some(dir) = self.s.image_path.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(dir);
        }
        let Some(mut path) = dialog.save_file() else {
            return;
        };
        if path.extension().is_none() {
            path.set_extension("stl");
        }
        let input = self.build_input();
        let res = self.s.resolution;
        let separate = self.s.separate_files;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let m = model::build_model(&input, res);
            let result = model::export(&m, &path, separate)
                .map(|files| {
                    let names: Vec<String> = files.iter().map(|p| file_name(p)).collect();
                    format!(
                        "Exported {} triangles in {:.1} s: {}",
                        m.triangle_count(),
                        t0.elapsed().as_secs_f32(),
                        names.join(", ")
                    )
                })
                .map_err(|e| format!("Export failed: {e}"));
            tx.send(result).ok();
        });
        self.export_job = Some(rx);
        self.set_status("Exporting…", false);
    }

    // ------------------------------------------------------------------ UI

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let mut cmd = None;
        ui.horizontal(|ui| {
            ui.heading("Lithophane Creator");
            ui.separator();
            ui.menu_button("Project", |ui| {
                let item = |ui: &mut egui::Ui, label: &str, shortcut: &KeyboardShortcut| {
                    let text = ui.ctx().format_shortcut(shortcut);
                    ui.add(egui::Button::new(label).shortcut_text(text))
                        .clicked()
                };
                if item(ui, "New", &SHORTCUT_NEW) {
                    cmd = Some(Command::New);
                }
                if item(ui, "Open…", &SHORTCUT_OPEN) {
                    cmd = Some(Command::Open);
                }
                ui.add_enabled_ui(!self.recent.is_empty(), |ui| {
                    ui.menu_button("Open recent", |ui| {
                        for p in &self.recent {
                            let resp = ui.button(file_name(p));
                            if resp.on_hover_text(p.display().to_string()).clicked() {
                                cmd = Some(Command::OpenPath(p.clone()));
                            }
                        }
                    });
                });
                ui.separator();
                if item(ui, "Save", &SHORTCUT_SAVE) {
                    cmd = Some(Command::Save);
                }
                if item(ui, "Save as…", &SHORTCUT_SAVE_AS) {
                    cmd = Some(Command::SaveAs);
                }
                if cmd.is_some() {
                    ui.close();
                }
            });
            ui.separator();
            if ui.button("Open image…").clicked() {
                self.open_image_dialog();
            }
            if ui.button("Open outline STL…").clicked() {
                self.open_stl_dialog();
            }
            ui.separator();
            let exporting = self.export_job.is_some();
            if ui
                .add_enabled(
                    !exporting && !self.shape.is_empty(),
                    egui::Button::new("Export STL…"),
                )
                .clicked()
            {
                self.start_export();
            }
            if exporting || self.job.is_some() {
                ui.add(egui::Spinner::new());
            }
            if let Some((msg, err)) = &self.status {
                let color = if *err {
                    Color32::from_rgb(255, 110, 110)
                } else {
                    ui.visuals().weak_text_color()
                };
                ui.colored_label(color, msg);
            }
        });
        if let Some(cmd) = cmd {
            self.run(cmd);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.spacing_mut().slider_width = 150.0;
            egui::CollapsingHeader::new("Outline")
                .default_open(true)
                .show(ui, |ui| self.outline_controls(ui));
            egui::CollapsingHeader::new("Image")
                .default_open(true)
                .show(ui, |ui| self.image_controls(ui));
            egui::CollapsingHeader::new("Lithophane")
                .default_open(true)
                .show(ui, |ui| self.litho_controls(ui));
            egui::CollapsingHeader::new("Supports")
                .default_open(true)
                .show(ui, |ui| self.support_controls(ui));
            egui::CollapsingHeader::new("Export")
                .default_open(true)
                .show(ui, |ui| self.export_controls(ui));
            ui.add_space(8.0);
            if ui.button("Reset all settings").clicked() {
                let image_path = self.s.image_path.clone();
                let stl_path = self.s.shape.stl_path.clone();
                self.s = Settings::default();
                self.s.image_path = image_path;
                self.s.shape.stl_path = stl_path;
                self.apply_image_orientation();
            }
        });
    }

    fn outline_controls(&mut self, ui: &mut egui::Ui) {
        let sh = &mut self.s.shape;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut sh.kind, ShapeKind::Rectangle, "Rectangle");
            ui.selectable_value(&mut sh.kind, ShapeKind::Ellipse, "Ellipse");
            ui.add_enabled_ui(self.stl.is_some(), |ui| {
                ui.selectable_value(&mut sh.kind, ShapeKind::Stl, "STL outline");
            });
        });
        egui::Grid::new("outline_grid")
            .num_columns(2)
            .show(ui, |ui| {
                match sh.kind {
                    ShapeKind::Rectangle | ShapeKind::Ellipse => {
                        ui.label("Width");
                        ui.add(
                            egui::DragValue::new(&mut sh.width)
                                .range(5.0..=1000.0)
                                .speed(0.5)
                                .suffix(" mm"),
                        );
                        ui.end_row();
                        ui.label("Height");
                        ui.add(
                            egui::DragValue::new(&mut sh.height)
                                .range(5.0..=1000.0)
                                .speed(0.5)
                                .suffix(" mm"),
                        );
                        ui.end_row();
                        if sh.kind == ShapeKind::Rectangle {
                            ui.label("Corner radius");
                            ui.add(
                                egui::DragValue::new(&mut sh.corner_radius)
                                    .range(0.0..=200.0)
                                    .speed(0.2)
                                    .suffix(" mm"),
                            );
                            ui.end_row();
                        }
                    }
                    ShapeKind::Stl => {
                        ui.label("File");
                        ui.label(sh.stl_path.as_deref().map(file_name).unwrap_or_default());
                        ui.end_row();
                        ui.label("Extrusion axis");
                        let axis_name = |a: usize| ["X", "Y", "Z"][a.min(2)];
                        egui::ComboBox::from_id_salt("axis")
                            .selected_text(match sh.axis {
                                AxisChoice::Auto => format!(
                                    "Auto ({})",
                                    self.detected_axis.map(axis_name).unwrap_or("?")
                                ),
                                a => format!("{a:?}"),
                            })
                            .show_ui(ui, |ui| {
                                for a in [
                                    AxisChoice::Auto,
                                    AxisChoice::X,
                                    AxisChoice::Y,
                                    AxisChoice::Z,
                                ] {
                                    ui.selectable_value(&mut sh.axis, a, format!("{a:?}"));
                                }
                            });
                        ui.end_row();
                        ui.label("Scale");
                        ui.add(
                            egui::DragValue::new(&mut sh.scale)
                                .range(0.01..=100.0)
                                .speed(0.01),
                        );
                        ui.end_row();
                    }
                }
                ui.label("Orientation");
                ui.horizontal(|ui| {
                    if ui.button("Rotate left").clicked() {
                        sh.quarter_turns = (sh.quarter_turns + 1) % 4;
                    }
                    if ui.button("Rotate right").clicked() {
                        sh.quarter_turns = (sh.quarter_turns + 3) % 4;
                    }
                    ui.checkbox(&mut sh.mirror, "Mirror");
                });
                ui.end_row();
            });
        let size = self.shape.size();
        ui.label(format!("Lithophane size: {:.1} × {:.1} mm", size.x, size.y));
        if let Some(e) = &self.shape_error {
            ui.colored_label(Color32::from_rgb(255, 110, 110), e);
        }
        if sh.kind == ShapeKind::Stl || self.stl.is_none() {
            ui.small("The STL's outline (cross-section of an extruded polygon) is the maximum boundary of the lithophane.");
        }
    }

    fn image_controls(&mut self, ui: &mut egui::Ui) {
        match &self.s.image_path {
            Some(p) if self.original_image.is_some() => {
                ui.label(format!(
                    "{} ({} × {} px)",
                    file_name(p),
                    self.image.w,
                    self.image.h
                ));
            }
            _ => {
                ui.label("No image loaded – showing a test pattern.");
            }
        }
        ui.horizontal(|ui| {
            if ui.button("Rotate 90°").clicked() {
                self.s.image_quarter_turns = (self.s.image_quarter_turns + 1) % 4;
                self.apply_image_orientation();
            }
            ui.checkbox(&mut self.s.litho.mirror_image, "Mirror");
            ui.checkbox(&mut self.s.adjustments.invert, "Invert");
        });
        self.histogram_ui(ui);
        let a = &mut self.s.adjustments;
        ui.add(egui::Slider::new(&mut a.lightness, -1.0..=1.0).text("Lightness"));
        ui.add(egui::Slider::new(&mut a.contrast, -0.9..=0.9).text("Contrast"));
        ui.add(
            egui::Slider::new(&mut a.gamma, 0.2..=5.0)
                .logarithmic(true)
                .text("Gamma"),
        );
        ui.add(egui::Slider::new(&mut a.black_point, 0.0..=1.0).text("Black point"));
        ui.add(egui::Slider::new(&mut a.white_point, 0.0..=1.0).text("White point"));
        if a.white_point <= a.black_point + 0.01 {
            a.white_point = (a.black_point + 0.01).min(1.0);
            a.black_point = a.white_point - 0.01;
        }
        if ui.button("Reset adjustments").clicked() {
            *a = Adjustments {
                invert: a.invert,
                ..Adjustments::default()
            };
        }
        ui.separator();
        let l = &mut self.s.litho;
        egui::Grid::new("fit_grid").num_columns(2).show(ui, |ui| {
            ui.label("Fit");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut l.fit, Fit::Cover, "Cover");
                ui.selectable_value(&mut l.fit, Fit::Contain, "Contain");
                ui.selectable_value(&mut l.fit, Fit::Stretch, "Stretch");
            });
            ui.end_row();
            ui.label("Zoom");
            ui.add(egui::Slider::new(&mut l.zoom, 0.1..=10.0).logarithmic(true));
            ui.end_row();
            ui.label("Offset X");
            ui.add(egui::Slider::new(&mut l.offset[0], -1.0..=1.0));
            ui.end_row();
            ui.label("Offset Y");
            ui.add(egui::Slider::new(&mut l.offset[1], -1.0..=1.0));
            ui.end_row();
        });
        if ui.button("Reset framing").clicked() {
            l.zoom = 1.0;
            l.offset = [0.0, 0.0];
        }
        ui.small(
            "Tip: drag the image in the Framing view to move it, scroll to zoom. \
             The picture reads correctly from the relief side; mirror it to view it from the flat back.",
        );
    }

    fn histogram_ui(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width().min(300.0), 48.0),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_gray(25));
        if self.histogram.is_empty() {
            return;
        }
        let max = self.histogram.iter().copied().max().unwrap_or(1).max(1) as f32;
        let bw = rect.width() / self.histogram.len() as f32;
        for (k, &v) in self.histogram.iter().enumerate() {
            let h = (v as f32 / max).sqrt() * (rect.height() - 2.0);
            let x = rect.left() + k as f32 * bw;
            let shade = (k as f32 / self.histogram.len() as f32 * 200.0) as u8 + 40;
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(x, rect.bottom() - h),
                    Pos2::new(x + bw - 0.5, rect.bottom()),
                ),
                0.0,
                Color32::from_gray(shade),
            );
        }
    }

    fn litho_controls(&mut self, ui: &mut egui::Ui) {
        let l = &mut self.s.litho;
        egui::Grid::new("litho_grid").num_columns(2).show(ui, |ui| {
            ui.label("Min thickness (white)");
            ui.add(
                egui::DragValue::new(&mut l.min_thickness)
                    .range(0.2..=10.0)
                    .speed(0.02)
                    .suffix(" mm"),
            );
            ui.end_row();
            ui.label("Max thickness (black)");
            ui.add(
                egui::DragValue::new(&mut l.max_thickness)
                    .range(0.5..=20.0)
                    .speed(0.02)
                    .suffix(" mm"),
            );
            ui.end_row();
            ui.label("Border width");
            ui.add(
                egui::DragValue::new(&mut l.border_width)
                    .range(0.0..=30.0)
                    .speed(0.05)
                    .suffix(" mm"),
            );
            ui.end_row();
            ui.label("Resolution (export)");
            ui.add(
                egui::DragValue::new(&mut self.s.resolution)
                    .range(0.05..=2.0)
                    .speed(0.005)
                    .suffix(" mm"),
            );
            ui.end_row();
        });
        if l.max_thickness < l.min_thickness + 0.1 {
            l.max_thickness = l.min_thickness + 0.1;
        }
        if let Some(p) = &self.preview {
            let factor = (p.res / self.s.resolution).powi(2);
            let tris = p.triangles as f64 * factor;
            ui.small(format!(
                "Preview: {:.2} mm grid, {} triangles, built in {} ms.\nExport: ~{:.1} M triangles (~{:.0} MB).",
                p.res,
                p.triangles,
                p.millis,
                tris / 1e6,
                tris * 50.0 / 1e6
            ));
        }
    }

    fn support_controls(&mut self, ui: &mut egui::Ui) {
        let sp = &mut self.s.supports;
        ui.checkbox(&mut sp.enabled, "Generate two support parts");
        ui.add_enabled_ui(sp.enabled, |ui| {
            egui::Grid::new("support_grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Wall position");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut sp.auto_position, "Auto");
                        ui.add_enabled(
                            !sp.auto_position,
                            egui::DragValue::new(&mut sp.inset)
                                .range(0.0..=500.0)
                                .speed(0.1)
                                .suffix(" mm inset"),
                        );
                    });
                    ui.end_row();
                    ui.label("Wall thickness");
                    ui.add(
                        egui::DragValue::new(&mut sp.wall_thickness)
                            .range(0.4..=5.0)
                            .speed(0.01)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Wall depth (bed)");
                    ui.add(egui::Slider::new(&mut sp.depth_ratio, 0.05..=0.6).text("× height"));
                    ui.end_row();
                    ui.label("Wall depth (top)");
                    ui.add(
                        egui::DragValue::new(&mut sp.top_width)
                            .range(1.0..=50.0)
                            .speed(0.05)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Contact gap");
                    ui.add(
                        egui::DragValue::new(&mut sp.contact_gap)
                            .range(0.0..=0.5)
                            .speed(0.001)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Foot thickness");
                    ui.add(
                        egui::DragValue::new(&mut sp.foot_thickness)
                            .range(0.1..=3.0)
                            .speed(0.01)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Foot width");
                    ui.add(
                        egui::DragValue::new(&mut sp.foot_width)
                            .range(5.0..=300.0)
                            .speed(0.2)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Foot overhang");
                    ui.add(
                        egui::DragValue::new(&mut sp.foot_extra)
                            .range(0.0..=100.0)
                            .speed(0.1)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Foot gap");
                    ui.add(
                        egui::DragValue::new(&mut sp.foot_gap)
                            .range(0.2..=10.0)
                            .speed(0.02)
                            .suffix(" mm"),
                    );
                    ui.end_row();
                    ui.label("Ribs & fillets");
                    ui.checkbox(&mut sp.ribs, "");
                    ui.end_row();
                    ui.label("Cradle under curved bottom");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut sp.cradle, "");
                        ui.add_enabled(
                            sp.cradle,
                            egui::DragValue::new(&mut sp.cradle_gap)
                                .range(0.05..=2.0)
                                .speed(0.01)
                                .suffix(" mm gap"),
                        );
                    });
                    ui.end_row();
                });
            if ui.button("Reset supports").clicked() {
                *sp = SupportParams::default();
            }
        });
    }

    fn export_controls(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.s.separate_files,
            "Also save each part as its own STL",
        );
        ui.small(
            "The STL holds three bodies: the lithophane and the left and right support parts. \
             Print standing up as exported; snap the supports off afterwards.",
        );
    }

    fn central(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Framing, "Framing");
            ui.selectable_value(&mut self.tab, Tab::Backlit, "Backlit preview");
            ui.selectable_value(&mut self.tab, Tab::Model, "3D model");
            if self.tab == Tab::Model {
                ui.separator();
                ui.checkbox(&mut self.backlit_3d, "Simulate backlight");
                if ui.button("Reset view").clicked() {
                    self.reset_camera();
                }
            }
        });
        ui.separator();
        match self.tab {
            Tab::Framing => self.framing_view(ui),
            Tab::Backlit => self.backlit_view(ui),
            Tab::Model => self.model_view(ui),
        }
    }

    fn framing_view(&mut self, ui: &mut egui::Ui) {
        ui.small("Drag to move the image, scroll to zoom. Blue: support contact strips (made full thickness).");
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(30));
        let size = self.shape.size();
        if size.x <= 0.0 || size.y <= 0.0 {
            return;
        }
        let scale = ((rect.width() - 40.0) as f64 / size.x)
            .min((rect.height() - 40.0) as f64 / size.y)
            .max(1e-3);
        let origin = rect.center()
            + Vec2::new(
                -(size.x * scale) as f32 * 0.5,
                (size.y * scale) as f32 * 0.5,
            );
        let to_screen = |p: DVec2| {
            Pos2::new(
                origin.x + (p.x * scale) as f32,
                origin.y - (p.y * scale) as f32,
            )
        };

        let map = ImageMap::new(size, self.image.w, self.image.h, &self.s.litho);
        if let Some(tex) = &self.image_tex {
            let (lo, hi) = map.image_rect();
            let r = Rect::from_two_pos(to_screen(lo), to_screen(hi));
            let uv = if self.s.litho.mirror_image {
                Rect::from_min_max(Pos2::new(1.0, 0.0), Pos2::new(0.0, 1.0))
            } else {
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0))
            };
            painter.image(tex.id(), r, uv, Color32::WHITE);
        }

        // Darken everything outside the outline.
        self.ensure_overlay();
        if let Some((_, pts, tris)) = &self.overlay {
            let mut mesh = egui::Mesh::default();
            let shade = Color32::from_black_alpha(215);
            for p in pts {
                mesh.colored_vertex(to_screen(*p), shade);
            }
            for t in tris {
                mesh.add_triangle(t[0], t[1], t[2]);
            }
            painter.add(egui::Shape::mesh(mesh));
        }
        let outline = Stroke::new(1.5, Color32::from_rgb(255, 200, 60));
        for lp in &self.shape.loops {
            painter.add(egui::Shape::closed_line(
                lp.iter().map(|p| to_screen(*p)).collect(),
                outline,
            ));
        }
        // Border rim and support contact positions.
        if let Some(plan) = crate::supports::plan(&self.shape, &self.s.supports) {
            for s in plan {
                let r = Rect::from_two_pos(
                    to_screen(DVec2::new(s.x0, 0.0)),
                    to_screen(DVec2::new(s.x1, s.height)),
                );
                painter.rect_filled(
                    r.expand2(Vec2::new(0.5, 0.0)),
                    0.0,
                    Color32::from_rgba_unmultiplied(90, 160, 255, 160),
                );
            }
        }

        // Drag to move the image, scroll to zoom.
        if resp.dragged() {
            let d = resp.drag_delta();
            let l = &mut self.s.litho;
            let dir = if l.mirror_image { -1.0 } else { 1.0 };
            l.offset[0] = (l.offset[0] + dir * d.x as f64 / (scale * size.x)).clamp(-1.0, 1.0);
            l.offset[1] = (l.offset[1] - d.y as f64 / (scale * size.y)).clamp(-1.0, 1.0);
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.s.litho.zoom =
                    (self.s.litho.zoom * (scroll as f64 * 0.002).exp()).clamp(0.1, 10.0);
            }
        }
    }

    /// Triangulated "everything except the outline" region for the framing view.
    fn ensure_overlay(&mut self) {
        if self.overlay.as_ref().is_some_and(|o| o.0 == self.shape_gen) {
            return;
        }
        let size = self.shape.size();
        let m = size.max_element() * 4.0;
        let mut loops = vec![vec![
            DVec2::new(-m, -m),
            DVec2::new(size.x + m, -m),
            DVec2::new(size.x + m, size.y + m),
            DVec2::new(-m, size.y + m),
        ]];
        loops.extend(self.shape.loops.iter().cloned());
        let mut pts: Vec<DVec2> = Vec::new();
        let mut tris = Vec::new();
        for (outer, holes) in group_loops(&loops) {
            let mut add = |li: usize, ccw: bool| -> Vec<u32> {
                let mut lp = loops[li].clone();
                if (crate::mesh::signed_area(&lp) > 0.0) != ccw {
                    lp.reverse();
                }
                let base = pts.len() as u32;
                pts.extend(lp.iter().copied());
                (0..lp.len() as u32).map(|k| base + k).collect()
            };
            let o = add(outer, true);
            let hs: Vec<Vec<u32>> = holes.iter().map(|&h| add(h, false)).collect();
            match triangulate(&pts, &o, &hs) {
                Some((t, steiner)) => {
                    pts.extend(steiner);
                    tris.extend(t);
                }
                None => tris.extend(earcut_raw(&pts, &o, &hs)),
            }
        }
        self.overlay = Some((self.shape_gen, pts, tris));
    }

    fn backlit_view(&mut self, ui: &mut egui::Ui) {
        ui.small("Simulated view with a light behind the lithophane (at preview resolution).");
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(12));
        if let Some(tex) = &self.backlit_tex {
            let ts = tex.size_vec2();
            let s = ((rect.width() - 20.0) / ts.x)
                .min((rect.height() - 20.0) / ts.y)
                .max(0.01);
            let r = Rect::from_center_size(rect.center(), ts * s);
            painter.image(
                tex.id(),
                r,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }

    fn model_view(&mut self, ui: &mut egui::Ui) {
        ui.small("Left drag: rotate · right/middle drag: pan · scroll: zoom · double-click: fit. Blue: support parts.");
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        ui.painter_at(rect)
            .rect_filled(rect, 0.0, Color32::from_rgb(38, 40, 46));
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.camera.orbit(d.x, d.y);
        }
        if resp.dragged_by(egui::PointerButton::Secondary)
            || resp.dragged_by(egui::PointerButton::Middle)
        {
            let d = resp.drag_delta();
            self.camera.pan(d.x, d.y, rect.height());
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll);
            }
        }
        if resp.double_clicked() {
            self.reset_camera();
        }
        let uniforms = Uniforms {
            view_proj: self.camera.view_proj(rect.width() / rect.height().max(1.0)),
            eye: self.camera.eye(),
            backlit: self.backlit_3d,
            t_range: [
                self.s.litho.min_thickness as f32,
                self.s.litho.max_thickness as f32,
            ],
        };
        let scene = self.scene.clone();
        let callback = egui_glow::CallbackFn::new(move |_info, painter| {
            scene.lock().unwrap().paint(painter.gl(), &uniforms);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(callback),
        });
        let painter = ui.painter_at(rect);
        if let Some(err) = &self.scene.lock().unwrap().error {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                err,
                egui::FontId::proportional(14.0),
                Color32::RED,
            );
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);
        self.handle_shortcuts(&ctx);
        self.poll_jobs(&ctx);

        egui::Panel::top("top_bar").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(380.0)
            .show(ui, |ui| {
                self.controls(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| self.central(ui));
        self.unsaved_changes_dialog(&ctx);

        let title = format!(
            "{}{} – Lithophane Creator",
            self.project_name(),
            if self.is_dirty() { "*" } else { "" }
        );
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }

        self.update_shape();
        self.update_image_texture(&ctx);
        self.start_preview_if_needed(&ctx);
        if self.job.is_some() || self.export_job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.s);
        eframe::set_value(storage, KEY_PROJECT, &self.project_path);
        eframe::set_value(storage, KEY_RECENT, &self.recent);
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.scene.lock().unwrap().destroy(gl);
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}
