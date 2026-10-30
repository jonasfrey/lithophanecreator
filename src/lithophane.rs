//! Heightmap sampling and meshing of the lithophane plate.
//!
//! The plate stands upright: X is the image's horizontal axis, Z is up and the
//! thickness goes along -Y. The back (y = 0) is flat, the relief faces -Y.
//! The outline is clipped exactly with marching squares, so arbitrary shapes
//! (with holes) produce a closed, watertight solid.

use glam::{DVec2, DVec3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::imaging::GrayImage;
use crate::mesh::{Mesh, group_loops, signed_area, triangulate};
use crate::shape::{EdgeIndex, Shape};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Fit {
    /// Fill the whole outline, cropping the image.
    Cover,
    /// Show the whole image; the rest stays at minimum thickness.
    Contain,
    /// Stretch the image to the outline's bounding box.
    Stretch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LithoParams {
    pub min_thickness: f64,
    pub max_thickness: f64,
    /// Solid rim along the outline, at maximum thickness.
    pub border_width: f64,
    pub fit: Fit,
    pub zoom: f64,
    /// Image centre offset as a fraction of the outline size.
    pub offset: [f64; 2],
    pub mirror_image: bool,
}

impl Default for LithoParams {
    fn default() -> Self {
        LithoParams {
            min_thickness: 0.8,
            max_thickness: 3.2,
            border_width: 2.0,
            fit: Fit::Cover,
            zoom: 1.0,
            offset: [0.0, 0.0],
            mirror_image: false,
        }
    }
}

/// Maps lithophane millimetres to image pixels.
#[derive(Clone, Copy, Debug)]
pub struct ImageMap {
    center: DVec2,
    /// Millimetres per pixel.
    scale: DVec2,
    img_size: DVec2,
    mirror: bool,
}

impl ImageMap {
    pub fn new(shape_size: DVec2, img_w: usize, img_h: usize, p: &LithoParams) -> ImageMap {
        let img = DVec2::new(img_w as f64, img_h as f64);
        let fit = shape_size / img;
        let scale = match p.fit {
            Fit::Cover => DVec2::splat(fit.max_element()),
            Fit::Contain => DVec2::splat(fit.min_element()),
            Fit::Stretch => fit,
        } / p.zoom.max(1e-3);
        let center = shape_size * (DVec2::splat(0.5) + DVec2::from(p.offset));
        ImageMap {
            center,
            scale,
            img_size: img,
            mirror: p.mirror_image,
        }
    }

    /// Pixel coordinates for a point (x right, y up) in millimetres.
    pub fn pixel_of(&self, p: DVec2) -> DVec2 {
        let mut px = (p.x - self.center.x) / self.scale.x + self.img_size.x * 0.5;
        if self.mirror {
            px = self.img_size.x - px;
        }
        let py = (self.center.y - p.y) / self.scale.y + self.img_size.y * 0.5;
        DVec2::new(px, py)
    }

    /// The image rectangle in millimetres (min, max).
    pub fn image_rect(&self) -> (DVec2, DVec2) {
        let half = self.img_size * self.scale * 0.5;
        (self.center - half, self.center + half)
    }

    pub fn mm_per_pixel(&self) -> f64 {
        self.scale.min_element()
    }
}

/// Samples of thickness on a regular grid covering the outline.
pub struct Heightmap {
    pub nx: usize,
    pub nz: usize,
    pub x0: f64,
    pub z0: f64,
    pub res: f64,
    pub inside: Vec<bool>,
    pub thickness: Vec<f32>,
    pub min_thickness: f64,
    pub max_thickness: f64,
    row_cross: Vec<Vec<f64>>,
    col_cross: Vec<Vec<f64>>,
}

impl Heightmap {
    fn x(&self, i: usize) -> f64 {
        self.x0 + i as f64 * self.res
    }
    fn z(&self, j: usize) -> f64 {
        self.z0 + j as f64 * self.res
    }
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Simulated look of the lithophane in front of a light, as RGBA (top row first).
    pub fn backlit_rgba(&self) -> (usize, usize, Vec<u8>) {
        let span = (self.max_thickness - self.min_thickness).max(1e-6);
        let mut out = vec![0u8; self.nx * self.nz * 4];
        for r in 0..self.nz {
            let j = self.nz - 1 - r;
            for i in 0..self.nx {
                let k = self.idx(i, j);
                if !self.inside[k] {
                    continue;
                }
                let t = (self.thickness[k] as f64 - self.min_thickness) / span;
                let c = transmission(t);
                let o = (r * self.nx + i) * 4;
                out[o] = (255.0 * c) as u8;
                out[o + 1] = (240.0 * c) as u8;
                out[o + 2] = (210.0 * c) as u8;
                out[o + 3] = 255;
            }
        }
        (self.nx, self.nz, out)
    }
}

/// Display brightness of light passing through material; `t` is the normalized
/// thickness (0 = thinnest). Light falls off exponentially with thickness.
pub fn transmission(t: f64) -> f64 {
    let linear = (-2.5 * t.clamp(0.0, 1.0)).exp();
    linear.powf(1.0 / 2.2)
}

fn prefilter_factor(shape: &Shape, image: &GrayImage, p: &LithoParams, res: f64) -> usize {
    let map = ImageMap::new(shape.size(), image.w, image.h, p);
    ((res.max(0.02) / 1.5) / map.mm_per_pixel())
        .floor()
        .max(1.0) as usize
}

/// The image box-filtered down to roughly the grid resolution (avoids aliasing and
/// keeps work proportional to the grid, not the photo).
pub fn prefiltered(shape: &Shape, image: &GrayImage, p: &LithoParams, res: f64) -> GrayImage {
    image.downscaled_by(prefilter_factor(shape, image, p, res))
}

/// Samples the image into a thickness grid over the outline.
/// `strips` are x-ranges forced to full thickness (support contact areas).
pub fn build_heightmap(
    shape: &Shape,
    image: &GrayImage,
    p: &LithoParams,
    res: f64,
    strips: &[(f64, f64)],
) -> Heightmap {
    let size = shape.size();
    let res = res.max(0.02);
    let jitter = res * 1.234e-3;
    let ncx = (size.x / res).ceil() as usize + 1;
    let ncz = (size.y / res).ceil() as usize + 1;
    let (nx, nz) = (ncx + 1, ncz + 1);
    let x0 = (size.x - ncx as f64 * res) * 0.5 + jitter;
    let z0 = (size.y - ncz as f64 * res) * 0.5 + jitter * 0.77;

    let small;
    let image = if prefilter_factor(shape, image, p, res) >= 2 {
        small = prefiltered(shape, image, p, res);
        &small
    } else {
        image
    };
    let map = ImageMap::new(size, image.w, image.h, p);

    let row_cross: Vec<Vec<f64>> = (0..nz)
        .into_par_iter()
        .map(|j| {
            let mut c = Vec::new();
            shape.crossings_h(z0 + j as f64 * res, &mut c);
            c
        })
        .collect();
    let col_cross: Vec<Vec<f64>> = (0..nx)
        .into_par_iter()
        .map(|i| {
            let mut c = Vec::new();
            shape.crossings_v(x0 + i as f64 * res, &mut c);
            c
        })
        .collect();

    let edge_index = (p.border_width > 0.0).then(|| EdgeIndex::new(shape, p.border_width));
    let span = p.max_thickness - p.min_thickness;
    let rows: Vec<(Vec<bool>, Vec<f32>)> = (0..nz)
        .into_par_iter()
        .map(|j| {
            let z = z0 + j as f64 * res;
            let cross = &row_cross[j];
            let mut inside = Vec::with_capacity(nx);
            let mut thick = Vec::with_capacity(nx);
            let mut k = 0;
            for i in 0..nx {
                let x = x0 + i as f64 * res;
                while k < cross.len() && cross[k] < x {
                    k += 1;
                }
                inside.push(k % 2 == 1);
                let pt = DVec2::new(x, z);
                let full = edge_index
                    .as_ref()
                    .is_some_and(|e| e.distance(pt) < p.border_width)
                    || strips.iter().any(|&(a, b)| x >= a && x <= b);
                let t = if full {
                    p.max_thickness
                } else {
                    let px = map.pixel_of(pt);
                    let lum = if px.x < 0.0
                        || px.y < 0.0
                        || px.x > image.w as f64
                        || px.y > image.h as f64
                    {
                        1.0
                    } else {
                        image.sample(px.x, px.y) as f64
                    };
                    p.min_thickness + (1.0 - lum) * span
                };
                thick.push(t as f32);
            }
            (inside, thick)
        })
        .collect();
    let mut inside = Vec::with_capacity(nx * nz);
    let mut thickness = Vec::with_capacity(nx * nz);
    for (a, b) in rows {
        inside.extend(a);
        thickness.extend(b);
    }
    Heightmap {
        nx,
        nz,
        x0,
        z0,
        res,
        inside,
        thickness,
        min_thickness: p.min_thickness,
        max_thickness: p.max_thickness,
        row_cross,
        col_cross,
    }
}

#[derive(Clone, Copy)]
enum Pt {
    /// Grid corner (i, j).
    C(usize, usize),
    /// Crossing on the horizontal grid edge (i, j)-(i+1, j).
    H(usize, usize),
    /// Crossing on the vertical grid edge (i, j)-(i, j+1).
    V(usize, usize),
}

impl Pt {
    fn is_crossing(self) -> bool {
        !matches!(self, Pt::C(..))
    }
}

struct Builder<'a> {
    hm: &'a Heightmap,
    mesh: Mesh,
    uv: Vec<DVec2>,
    corner_id: Vec<u32>,
    h_id: Vec<u32>,
    v_id: Vec<u32>,
}

impl Builder<'_> {
    fn add(&mut self, x: f64, z: f64, t: f64) -> u32 {
        self.uv.push(DVec2::new(x, z));
        self.mesh.attr.push(t as f32);
        self.mesh.add_vertex(DVec3::new(x, -t, z))
    }

    /// First crossing in [a, b), clamped away from the ends.
    fn crossing_in(list: &[f64], a: f64, b: f64) -> f64 {
        let lo = list.partition_point(|&c| c < a);
        let hi = list.partition_point(|&c| c < b);
        let c = if hi > lo && (hi - lo) % 2 == 1 {
            list[lo]
        } else {
            0.5 * (a + b)
        };
        let eps = (b - a) * 1e-3;
        c.clamp(a + eps, b - eps)
    }

    fn id(&mut self, p: Pt) -> u32 {
        let hm = self.hm;
        match p {
            Pt::C(i, j) => {
                let k = hm.idx(i, j);
                if self.corner_id[k] == u32::MAX {
                    self.corner_id[k] = self.add(hm.x(i), hm.z(j), hm.thickness[k] as f64);
                }
                self.corner_id[k]
            }
            Pt::H(i, j) => {
                let k = hm.idx(i, j);
                if self.h_id[k] == u32::MAX {
                    let (a, b) = (hm.x(i), hm.x(i + 1));
                    let x = Self::crossing_in(&hm.row_cross[j], a, b);
                    let f = (x - a) / (b - a);
                    let t = hm.thickness[k] as f64 * (1.0 - f)
                        + hm.thickness[hm.idx(i + 1, j)] as f64 * f;
                    self.h_id[k] = self.add(x, hm.z(j), t);
                }
                self.h_id[k]
            }
            Pt::V(i, j) => {
                let k = hm.idx(i, j);
                if self.v_id[k] == u32::MAX {
                    let (a, b) = (hm.z(j), hm.z(j + 1));
                    let z = Self::crossing_in(&hm.col_cross[i], a, b);
                    let f = (z - a) / (b - a);
                    let t = hm.thickness[k] as f64 * (1.0 - f)
                        + hm.thickness[hm.idx(i, j + 1)] as f64 * f;
                    self.v_id[k] = self.add(hm.x(i), z, t);
                }
                self.v_id[k]
            }
        }
    }
}

/// Builds the closed lithophane solid from a heightmap.
pub fn mesh_heightmap(hm: &Heightmap, shape: &Shape) -> Mesh {
    let n = hm.nx * hm.nz;
    let mut b = Builder {
        hm,
        mesh: Mesh::default(),
        uv: Vec::new(),
        corner_id: vec![u32::MAX; n],
        h_id: vec![u32::MAX; n],
        v_id: vec![u32::MAX; n],
    };
    let mut segments: Vec<(u32, u32)> = Vec::new();
    let mut ring: Vec<Pt> = Vec::with_capacity(8);
    let mut ids: Vec<u32> = Vec::with_capacity(8);

    for j in 0..hm.nz - 1 {
        for i in 0..hm.nx - 1 {
            let corners = [
                Pt::C(i, j),
                Pt::C(i + 1, j),
                Pt::C(i + 1, j + 1),
                Pt::C(i, j + 1),
            ];
            let edges = [Pt::H(i, j), Pt::V(i + 1, j), Pt::H(i, j + 1), Pt::V(i, j)];
            let inside = [
                hm.inside[hm.idx(i, j)],
                hm.inside[hm.idx(i + 1, j)],
                hm.inside[hm.idx(i + 1, j + 1)],
                hm.inside[hm.idx(i, j + 1)],
            ];
            let count = inside.iter().filter(|&&v| v).count();
            if count == 0 {
                continue;
            }
            if count == 4 {
                let c: Vec<u32> = corners.iter().map(|&p| b.id(p)).collect();
                b.mesh.triangles.push([c[0], c[1], c[2]]);
                b.mesh.triangles.push([c[0], c[2], c[3]]);
                continue;
            }
            let saddle = count == 2 && inside[0] == inside[2];
            let mut rings: Vec<Vec<Pt>> = Vec::new();
            if saddle {
                let center = DVec2::new(hm.x(i) + 0.5 * hm.res, hm.z(j) + 0.5 * hm.res);
                if !shape.contains(center) {
                    for k in (0..4).filter(|&k| inside[k]) {
                        rings.push(vec![edges[(k + 3) % 4], corners[k], edges[k]]);
                    }
                }
            }
            if rings.is_empty() {
                ring.clear();
                for k in 0..4 {
                    if inside[k] {
                        ring.push(corners[k]);
                    }
                    if inside[k] != inside[(k + 1) % 4] {
                        ring.push(edges[k]);
                    }
                }
                rings.push(ring.clone());
            }
            for r in &rings {
                ids.clear();
                for &p in r {
                    let id = b.id(p);
                    ids.push(id);
                }
                for m in 1..ids.len() - 1 {
                    b.mesh.triangles.push([ids[0], ids[m], ids[m + 1]]);
                }
                for m in 0..r.len() {
                    let m2 = (m + 1) % r.len();
                    if r[m].is_crossing() && r[m2].is_crossing() {
                        segments.push((ids[m], ids[m2]));
                    }
                }
            }
        }
    }

    let front_count = b.mesh.positions.len();
    let front_tris = b.mesh.triangles.len();
    let Builder { mut mesh, uv, .. } = b;

    // Chain the boundary into loops (inside on the left: outer CCW, holes CW).
    let mut next = vec![u32::MAX; front_count];
    for &(a, c) in &segments {
        next[a as usize] = c;
    }
    let mut visited = vec![false; front_count];
    let mut loops: Vec<Vec<u32>> = Vec::new();
    for &(start, _) in &segments {
        if visited[start as usize] {
            continue;
        }
        let mut lp = Vec::new();
        let mut cur = start;
        while !visited[cur as usize] && cur != u32::MAX {
            visited[cur as usize] = true;
            lp.push(cur);
            cur = next[cur as usize];
        }
        loops.push(lp);
    }

    // Flat back: triangulate the boundary loops; fall back to mirroring the front.
    let mut back_of = vec![u32::MAX; front_count];
    let mut back_tris: Vec<[u32; 3]> = Vec::new();
    let ok = (|| {
        let mut pts: Vec<DVec2> = Vec::new();
        let mut local: Vec<Vec<u32>> = Vec::new();
        let mut owner: Vec<u32> = Vec::new();
        for lp in &loops {
            local.push(
                lp.iter()
                    .map(|&f| {
                        pts.push(uv[f as usize]);
                        owner.push(f);
                        (pts.len() - 1) as u32
                    })
                    .collect(),
            );
        }
        let loops2d: Vec<Vec<DVec2>> = local
            .iter()
            .map(|l| l.iter().map(|&k| pts[k as usize]).collect())
            .collect();
        let mut tris = Vec::new();
        let mut steiner_all: Vec<DVec2> = Vec::new();
        for (o, holes) in group_loops(&loops2d) {
            if signed_area(&loops2d[o]) <= 0.0
                || holes.iter().any(|&h| signed_area(&loops2d[h]) >= 0.0)
            {
                return None;
            }
            let hole_refs: Vec<Vec<u32>> = holes.iter().map(|&h| local[h].clone()).collect();
            let mut all = pts.clone();
            all.extend_from_slice(&steiner_all);
            let (t, steiner) = triangulate(&all, &local[o], &hole_refs)?;
            steiner_all.extend(steiner);
            tris.extend(t);
        }
        Some((owner, pts.len(), steiner_all, tris))
    })();
    match ok {
        Some((owner, npts, steiner, tris)) => {
            let mut local_to_back = Vec::with_capacity(npts + steiner.len());
            for &f in &owner {
                let p = uv[f as usize];
                let id = mesh.add_vertex(DVec3::new(p.x, 0.0, p.y));
                mesh.attr.push(-1.0);
                back_of[f as usize] = id;
                local_to_back.push(id);
            }
            for p in steiner {
                local_to_back.push(mesh.add_vertex(DVec3::new(p.x, 0.0, p.y)));
                mesh.attr.push(-1.0);
            }
            for t in tris {
                let m = t.map(|k| local_to_back[k as usize]);
                back_tris.push([m[0], m[2], m[1]]);
            }
        }
        None => {
            for f in 0..front_count {
                let p = uv[f];
                back_of[f] = mesh.add_vertex(DVec3::new(p.x, 0.0, p.y));
                mesh.attr.push(-1.0);
            }
            for t in &mesh.triangles[..front_tris] {
                let m = t.map(|k| back_of[k as usize]);
                back_tris.push([m[0], m[2], m[1]]);
            }
        }
    }
    mesh.triangles.extend(back_tris);

    // Side walls along the outline.
    for &(a, c) in &segments {
        let (ba, bc) = (back_of[a as usize], back_of[c as usize]);
        mesh.triangles.push([a, ba, bc]);
        mesh.triangles.push([a, bc, c]);
    }
    mesh.orient_outward();
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> GrayImage {
        let data = (0..w * h).map(|k| (k % w) as f32 / w as f32).collect();
        GrayImage { w, h, data }
    }

    fn check(shape: &Shape, res: f64) -> Mesh {
        let p = LithoParams::default();
        let hm = build_heightmap(shape, &gradient(64, 48), &p, res, &[(10.0, 11.0)]);
        let m = mesh_heightmap(&hm, shape);
        assert_eq!(m.open_edge_count(), 0, "mesh must be watertight");
        assert_eq!(m.attr.len(), m.positions.len());
        let vol = m.signed_volume();
        let area = shape.area();
        assert!(
            vol > area * p.min_thickness * 0.95 && vol < area * p.max_thickness * 1.05,
            "volume {vol}"
        );
        // The flat back only needs the outline vertices (no fallback to a mirrored front).
        let back = m.attr.iter().filter(|&&a| a < 0.0).count();
        assert!(
            back * 3 < m.positions.len() - back,
            "back face fell back to the dense grid"
        );
        m
    }

    #[test]
    fn rectangle_is_watertight() {
        let m = check(&Shape::rectangle(40.0, 30.0, 0.0), 0.5);
        let (lo, hi) = m.bbox().unwrap();
        assert!((hi.x - lo.x - 40.0).abs() < 0.3 && (hi.z - lo.z - 30.0).abs() < 0.3);
        assert!((lo.y + 3.2).abs() < 1e-4 && hi.y.abs() < 1e-6);
    }

    #[test]
    fn ellipse_and_rounded_rect_are_watertight() {
        check(&Shape::ellipse(50.0, 35.0), 0.4);
        check(&Shape::rectangle(30.0, 30.0, 6.0), 0.3);
    }

    #[test]
    fn shape_with_holes_is_watertight() {
        let mut s = Shape::rectangle(40.0, 40.0, 0.0);
        let hole = Shape::ellipse(10.0, 10.0).loops.remove(0);
        s.loops
            .push(hole.iter().map(|p| *p + DVec2::new(5.0, 5.0)).collect());
        s.loops
            .push(hole.iter().map(|p| *p + DVec2::new(25.0, 22.0)).collect());
        // An island inside a hole.
        let island = Shape::ellipse(4.0, 4.0).loops.remove(0);
        s.loops
            .push(island.iter().map(|p| *p + DVec2::new(8.0, 8.0)).collect());
        check(&s, 0.35);
    }

    #[test]
    fn thin_features_and_saddles() {
        // A zig-zag with features near the grid resolution.
        let mut lp = Vec::new();
        for k in 0..20 {
            lp.push(DVec2::new(
                k as f64 * 2.0,
                if k % 2 == 0 { 0.0 } else { 1.2 },
            ));
        }
        lp.push(DVec2::new(40.0, 20.0));
        lp.push(DVec2::new(0.0, 20.0));
        check(&Shape { loops: vec![lp] }.normalized(), 0.5);
    }
}
