mod app;
mod imaging;
mod lithophane;
mod mesh;
mod model;
mod project;
mod shape;
mod supports;
mod viewer;

use std::path::{Path, PathBuf};
use std::sync::Arc;

const USAGE: &str = "\
Lithophane Creator

Without arguments the graphical application starts.
  lithophane-creator <project.lithoproj>   opens a saved project in the application

Headless export:
  lithophane-creator --out <file.stl> [options]

Options:
  --project <file>       start from a saved project (other options override it)
  --image <file>         picture (png, jpg, ...); a test pattern is used if omitted
  --outline <file.stl>   extruded-polygon STL that bounds the lithophane
  --width <mm>           width of the rectangle (without --outline), default 108
  --height <mm>          height of the rectangle (without --outline), default 144
  --ellipse              use an ellipse of width x height instead of a rectangle
  --res <mm>             relief grid spacing, default 0.2
  --min <mm> --max <mm>  thickness range, default 0.8 / 3.2
  --border <mm>          solid rim width, default 2
  --count <n>            number of lithophanes standing one behind the other, 1-10, default 1
  --spacing <mm>         gap between neighbouring lithophanes, default 3
  --no-supports          only export the lithophane
  --separate             also write one STL per part
  --preview <file.png>   also save a simulated backlit view
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => return run_gui(None),
        [p] if project::is_project_file(Path::new(p)) => return run_gui(Some(p.into())),
        _ => {}
    }
    if let Err(e) = run_cli(&args) {
        eprintln!("error: {e}\n\n{USAGE}");
        std::process::exit(1);
    }
}

fn run_gui(open: Option<PathBuf>) {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Lithophane Creator")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Glow,
        depth_buffer: 24,
        multisampling: 4,
        ..Default::default()
    };
    if let Err(e) = eframe::run_native(
        "Lithophane Creator",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, open)))),
    ) {
        eprintln!("failed to start the application: {e}");
        std::process::exit(1);
    }
}

fn run_cli(args: &[String]) -> Result<(), String> {
    let mut s = match args.iter().position(|a| a == "--project") {
        Some(i) => {
            let p = args.get(i + 1).ok_or("--project needs a value")?;
            project::load(Path::new(p))?
        }
        None => model::Settings::default(),
    };
    let mut out: Option<PathBuf> = None;
    let mut image_path = s.image_path.clone();
    let mut preview: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        let mut number = |name: &str| -> Result<f64, String> {
            value(name)?
                .parse()
                .map_err(|_| format!("{name} needs a number"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--project" => {
                value(a)?;
            }
            "--out" => out = Some(value(a)?.into()),
            "--image" => image_path = Some(value(a)?.into()),
            "--outline" => {
                s.shape.stl_path = Some(value(a)?.into());
                s.shape.kind = model::ShapeKind::Stl;
            }
            "--width" => {
                s.shape.width = number(a)?;
                if s.shape.kind == model::ShapeKind::Stl {
                    s.shape.kind = model::ShapeKind::Rectangle;
                }
            }
            "--height" => {
                s.shape.height = number(a)?;
                if s.shape.kind == model::ShapeKind::Stl {
                    s.shape.kind = model::ShapeKind::Rectangle;
                }
            }
            "--ellipse" => s.shape.kind = model::ShapeKind::Ellipse,
            "--res" => s.resolution = number(a)?,
            "--min" => s.litho.min_thickness = number(a)?,
            "--max" => s.litho.max_thickness = number(a)?,
            "--border" => s.litho.border_width = number(a)?,
            "--count" => {
                let n = number(a)?;
                if n.fract() != 0.0 || !(1.0..=model::MAX_COUNT as f64).contains(&n) {
                    return Err(format!("--count must be 1 to {}", model::MAX_COUNT));
                }
                s.layout.count = n as u32;
            }
            "--spacing" => s.layout.spacing = number(a)?,
            "--no-supports" => s.supports.enabled = false,
            "--separate" => s.separate_files = true,
            "--preview" => preview = Some(value(a)?.into()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let out = out.ok_or("--out is required for headless export")?;
    let stl = match &s.shape.stl_path {
        Some(p) if s.shape.kind == model::ShapeKind::Stl => Some(mesh::read_stl(p)?),
        _ => None,
    };
    let (shape, _) = model::make_shape(&s.shape, stl.as_deref())?;
    let mut image = match &image_path {
        Some(p) => imaging::GrayImage::load(p)?.downscaled(imaging::MAX_IMAGE_DIM),
        None => model::test_pattern(),
    };
    for _ in 0..s.image_quarter_turns % 4 {
        image = image.rotated_cw();
    }
    let size = shape.size();
    println!(
        "outline: {:.1} x {:.1} mm, {} loop(s)",
        size.x,
        size.y,
        shape.loops.len()
    );
    let input = model::BuildInput {
        shape: Arc::new(shape),
        image: Arc::new(image),
        adjustments: s.adjustments,
        litho: s.litho.clone(),
        supports: s.supports.clone(),
        layout: s.layout,
    };
    let t0 = std::time::Instant::now();
    let m = model::build_model(&input, s.resolution);
    let files = model::export(&m, &out, s.separate_files).map_err(|e| e.to_string())?;
    println!(
        "{} triangles in {:.2} s (lithophane watertight: {})",
        m.triangle_count(),
        t0.elapsed().as_secs_f32(),
        m.lithophane.open_edge_count() == 0
    );
    for f in files {
        println!("wrote {}", f.display());
    }
    if let Some(p) = preview {
        let (w, h, rgba) = m.heightmap.backlit_rgba();
        image::save_buffer(
            &p,
            &rgba,
            w as u32,
            h as u32,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| format!("cannot write {}: {e}", p.display()))?;
        println!("wrote {}", p.display());
    }
    Ok(())
}
