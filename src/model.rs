//! Settings and assembly of the printable model (lithophanes + two support parts).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::imaging::{Adjustments, GrayImage};
use crate::lithophane::{Heightmap, LithoParams, build_heightmap, mesh_heightmap};
use crate::mesh::{Mesh, write_stl};
use crate::shape::{AxisChoice, Shape};
use crate::supports::{self, SupportParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeKind {
    Rectangle,
    Ellipse,
    Stl,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShapeSettings {
    pub kind: ShapeKind,
    pub width: f64,
    pub height: f64,
    pub corner_radius: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stl_path: Option<PathBuf>,
    pub axis: AxisChoice,
    pub quarter_turns: u8,
    pub mirror: bool,
    pub scale: f64,
}

impl Default for ShapeSettings {
    fn default() -> Self {
        // Size of the example: a 104 x 140 mm picture plus a 2 mm rim.
        ShapeSettings {
            kind: ShapeKind::Rectangle,
            width: 108.0,
            height: 144.0,
            corner_radius: 0.0,
            stl_path: None,
            axis: AxisChoice::Auto,
            quarter_turns: 0,
            mirror: false,
            scale: 1.0,
        }
    }
}

pub const MAX_COUNT: u32 = 10;
const MIN_SPACING: f64 = 1.0;

/// Identical lithophanes standing one behind the other, sharing the two support parts.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layout {
    /// Number of lithophanes, 1 to `MAX_COUNT`.
    pub count: u32,
    /// Gap between the full-thickness faces of neighbouring lithophanes in mm.
    pub spacing: f64,
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            count: 1,
            spacing: 3.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub shape: ShapeSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_path: Option<PathBuf>,
    /// Clockwise quarter turns applied to the loaded image.
    pub image_quarter_turns: u8,
    pub adjustments: Adjustments,
    pub litho: LithoParams,
    pub supports: SupportParams,
    pub layout: Layout,
    /// Grid spacing of the exported relief in mm.
    pub resolution: f64,
    /// Also write each part to its own STL next to the combined one.
    pub separate_files: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            shape: ShapeSettings::default(),
            image_path: None,
            image_quarter_turns: 0,
            adjustments: Adjustments::default(),
            litho: LithoParams::default(),
            supports: SupportParams::default(),
            layout: Layout::default(),
            resolution: 0.2,
            separate_files: false,
        }
    }
}

/// Everything needed to build the model, cheap to clone and send to a worker thread.
#[derive(Clone)]
pub struct BuildInput {
    pub shape: Arc<Shape>,
    pub image: Arc<GrayImage>,
    pub adjustments: Adjustments,
    pub litho: LithoParams,
    pub supports: SupportParams,
    pub layout: Layout,
}

pub struct Model {
    /// The first lithophane; the others are copies of it.
    pub lithophane: Mesh,
    /// Offset of each lithophane (the first one's is zero).
    pub copies: Vec<DVec3>,
    pub supports: Vec<Mesh>,
    pub heightmap: Heightmap,
}

impl Model {
    pub fn triangle_count(&self) -> usize {
        self.lithophane.triangles.len() * self.copies.len()
            + self
                .supports
                .iter()
                .map(|m| m.triangles.len())
                .sum::<usize>()
    }

    /// Name, mesh and offset of every printed body.
    pub fn parts(&self) -> Vec<(String, &Mesh, DVec3)> {
        let mut v: Vec<(String, &Mesh, DVec3)> = self
            .copies
            .iter()
            .enumerate()
            .map(|(k, &at)| {
                let name = if self.copies.len() == 1 {
                    "lithophane".to_owned()
                } else {
                    format!("lithophane_{}", k + 1)
                };
                (name, &self.lithophane, at)
            })
            .collect();
        let names = ["support_left", "support_right"];
        for (k, m) in self.supports.iter().enumerate() {
            v.push((names[k.min(1)].to_owned(), m, DVec3::ZERO));
        }
        v
    }
}

pub fn build_model(input: &BuildInput, resolution: f64) -> Model {
    let shape = &*input.shape;
    let mut litho = input.litho.clone();
    litho.max_thickness = litho.max_thickness.max(litho.min_thickness + 0.05);
    let image = crate::lithophane::prefiltered(shape, &input.image, &litho, resolution)
        .adjusted(&input.adjustments);
    let plan = supports::plan(shape, &input.supports);
    let strips = supports::contact_strips(&plan);
    let heightmap = build_heightmap(shape, &image, &litho, resolution, &strips);
    let mut lithophane = mesh_heightmap(&heightmap, shape);
    let count = input.layout.count.clamp(1, MAX_COUNT) as usize;
    let spacing = input.layout.spacing.max(MIN_SPACING);
    let mut parts: Vec<Mesh> = match &plan {
        Some(plan) => supports::build(
            shape,
            plan,
            litho.max_thickness,
            count,
            spacing,
            &input.supports,
        )
        .into(),
        None => Vec::new(),
    };
    // The lithophanes stand one behind the other along +Y.
    let pitch = litho.max_thickness + spacing;
    let copies = (0..count)
        .map(|k| DVec3::new(0.0, k as f64 * pitch, 0.0))
        .collect();
    // Centre on the build plate, the row's mid-plane on y = 0.
    let size = shape.size();
    let depth = litho.max_thickness + (count - 1) as f64 * pitch;
    let shift = DVec3::new(-0.5 * size.x, litho.max_thickness - 0.5 * depth, 0.0);
    lithophane.translate(shift);
    for m in &mut parts {
        m.translate(shift);
    }
    Model {
        lithophane,
        copies,
        supports: parts,
        heightmap,
    }
}

/// Writes the combined STL (and optionally one file per part). Returns the written paths.
pub fn export(model: &Model, path: &Path, separate: bool) -> std::io::Result<Vec<PathBuf>> {
    let meshes: Vec<(&Mesh, DVec3)> = model
        .parts()
        .into_iter()
        .map(|(_, m, at)| (m, at))
        .collect();
    write_stl(path, &meshes)?;
    let mut written = vec![path.to_path_buf()];
    if separate {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lithophane");
        for (name, mesh, at) in model.parts() {
            let p = path.with_file_name(format!("{stem}_{name}.stl"));
            write_stl(&p, &[(mesh, at)])?;
            written.push(p);
        }
    }
    Ok(written)
}

/// The outline for the current settings. `stl` holds the loaded STL triangles.
pub fn make_shape(
    s: &ShapeSettings,
    stl: Option<&[[[f32; 3]; 3]]>,
) -> Result<(Shape, Option<usize>), String> {
    let (base, axis) = match s.kind {
        ShapeKind::Rectangle => (
            Shape::rectangle(s.width.max(1.0), s.height.max(1.0), s.corner_radius),
            None,
        ),
        ShapeKind::Ellipse => (Shape::ellipse(s.width.max(1.0), s.height.max(1.0)), None),
        ShapeKind::Stl => {
            let tris = stl.ok_or("no outline STL loaded")?;
            let (shape, axis) = crate::shape::outline_from_triangles(tris, s.axis)?;
            (shape, Some(axis))
        }
    };
    let scale = if s.kind == ShapeKind::Stl {
        s.scale.max(1e-3)
    } else {
        1.0
    };
    let shape = base.transformed(s.quarter_turns, s.mirror, scale);
    let size = shape.size();
    if size.x < 2.0 || size.y < 2.0 {
        return Err(format!(
            "outline is too small ({:.1} x {:.1} mm)",
            size.x, size.y
        ));
    }
    Ok((shape, axis))
}

/// Placeholder picture used until an image is loaded: soft gradient with a few discs.
pub fn test_pattern() -> GrayImage {
    let (w, h) = (256usize, 256usize);
    let mut data = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let mut c = 0.15 + 0.8 * v;
            for (cx, cy, r, val) in [
                (0.3, 0.3, 0.14, 0.95),
                (0.7, 0.45, 0.18, 0.1),
                (0.4, 0.75, 0.1, 0.5),
            ] {
                if (u - cx).powi(2) + (v - cy).powi(2) < r * r {
                    c = val;
                }
            }
            data[y * w + x] = c;
        }
    }
    GrayImage { w, h, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(settings: &Settings) -> BuildInput {
        let (shape, _) = make_shape(&settings.shape, None).unwrap();
        BuildInput {
            shape: Arc::new(shape),
            image: Arc::new(test_pattern()),
            adjustments: settings.adjustments,
            litho: settings.litho.clone(),
            supports: settings.supports.clone(),
            layout: settings.layout,
        }
    }

    #[test]
    fn full_model_for_example_dimensions() {
        let model = build_model(&input(&Settings::default()), 1.0);
        assert_eq!(model.lithophane.open_edge_count(), 0);
        assert_eq!(model.copies, [DVec3::ZERO]);
        assert_eq!(model.supports.len(), 2);
        let (lo, hi) = model.lithophane.bbox().unwrap();
        assert!((hi.x - lo.x - 108.0).abs() < 0.6);
        assert!((hi.z - lo.z - 144.0).abs() < 0.6);
        assert!(lo.z.abs() < 1e-6);
        // Supports must not overlap the plate's slab except through the contact gap.
        let (slo, shi) = model.supports[0].bbox().unwrap();
        assert!(slo.z.abs() < 1e-6 && shi.z <= hi.z + 1e-4);
    }

    #[test]
    fn several_lithophanes_stand_centred_behind_each_other() {
        let settings = Settings {
            layout: Layout {
                count: 3,
                spacing: 4.0,
            },
            ..Settings::default()
        };
        let model = build_model(&input(&settings), 1.0);
        let pitch = settings.litho.max_thickness + 4.0;
        assert_eq!(model.copies.len(), 3);
        assert!((model.copies[2].y - 2.0 * pitch).abs() < 1e-9);
        let names: Vec<String> = model.parts().into_iter().map(|(n, _, _)| n).collect();
        assert_eq!(
            names,
            [
                "lithophane_1",
                "lithophane_2",
                "lithophane_3",
                "support_left",
                "support_right"
            ]
        );
        let (lo, hi) = model.lithophane.bbox().unwrap();
        let (front, back) = (lo.y, hi.y + model.copies[2].y);
        assert!(
            (front + back).abs() < 1e-4,
            "row not centred: {front} .. {back}"
        );
        assert!((back - front - (settings.litho.max_thickness + 2.0 * pitch)).abs() < 1e-4);
        let single = build_model(&input(&Settings::default()), 1.0);
        assert_eq!(
            model.triangle_count()
                - model
                    .supports
                    .iter()
                    .map(|m| m.triangles.len())
                    .sum::<usize>(),
            3 * single.lithophane.triangles.len()
        );

        // The STL holds every copy at its place.
        let dir = std::env::temp_dir().join(format!("litho-model-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("row.stl");
        let files = export(&model, &path, true).unwrap();
        assert_eq!(files.len(), 6);
        assert!(files[3].ends_with("row_lithophane_3.stl"));
        let tris = crate::mesh::read_stl(&path).unwrap();
        assert_eq!(tris.len(), model.triangle_count());
        let max_y = tris
            .iter()
            .flatten()
            .map(|v| v[1] as f64)
            .fold(f64::MIN, f64::max);
        let (_, shi) = model.supports[0].bbox().unwrap();
        assert!((max_y - shi.y).abs() < 1e-3);
        std::fs::remove_dir_all(&dir).ok();
    }
}
