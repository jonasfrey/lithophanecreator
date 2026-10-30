//! Settings and assembly of the printable model (lithophane + two support parts).

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
}

pub struct Model {
    pub lithophane: Mesh,
    pub supports: Vec<Mesh>,
    pub heightmap: Heightmap,
}

impl Model {
    pub fn triangle_count(&self) -> usize {
        self.lithophane.triangles.len()
            + self
                .supports
                .iter()
                .map(|m| m.triangles.len())
                .sum::<usize>()
    }

    pub fn parts(&self) -> Vec<(&'static str, &Mesh)> {
        let mut v = vec![("lithophane", &self.lithophane)];
        let names = ["support_left", "support_right"];
        for (k, m) in self.supports.iter().enumerate() {
            v.push((names[k.min(1)], m));
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
    let mut parts: Vec<Mesh> = match &plan {
        Some(plan) => supports::build(shape, plan, litho.max_thickness, &input.supports).into(),
        None => Vec::new(),
    };
    // Centre on the build plate, plate's mid-plane on y = 0.
    let size = shape.size();
    let shift = DVec3::new(-0.5 * size.x, 0.5 * litho.max_thickness, 0.0);
    lithophane.translate(shift);
    for m in &mut parts {
        m.translate(shift);
    }
    Model {
        lithophane,
        supports: parts,
        heightmap,
    }
}

/// Writes the combined STL (and optionally one file per part). Returns the written paths.
pub fn export(model: &Model, path: &Path, separate: bool) -> std::io::Result<Vec<PathBuf>> {
    let meshes: Vec<&Mesh> = model.parts().into_iter().map(|(_, m)| m).collect();
    write_stl(path, &meshes)?;
    let mut written = vec![path.to_path_buf()];
    if separate {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lithophane");
        for (name, mesh) in model.parts() {
            let p = path.with_file_name(format!("{stem}_{name}.stl"));
            write_stl(&p, &[mesh])?;
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

    #[test]
    fn full_model_for_example_dimensions() {
        let settings = Settings::default();
        let (shape, _) = make_shape(&settings.shape, None).unwrap();
        let input = BuildInput {
            shape: Arc::new(shape),
            image: Arc::new(test_pattern()),
            adjustments: settings.adjustments,
            litho: settings.litho.clone(),
            supports: settings.supports.clone(),
        };
        let model = build_model(&input, 1.0);
        assert_eq!(model.lithophane.open_edge_count(), 0);
        assert_eq!(model.supports.len(), 2);
        let (lo, hi) = model.lithophane.bbox().unwrap();
        assert!((hi.x - lo.x - 108.0).abs() < 0.6);
        assert!((hi.z - lo.z - 144.0).abs() < 0.6);
        assert!(lo.z.abs() < 1e-6);
        // Supports must not overlap the plate's slab except through the contact gap.
        let (slo, shi) = model.supports[0].bbox().unwrap();
        assert!(slo.z.abs() < 1e-6 && shi.z <= hi.z + 1e-4);
    }
}
