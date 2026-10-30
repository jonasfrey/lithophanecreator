//! 2D outline of the lithophane: built-in shapes, or the cross-section of an
//! extruded-polygon STL. Loops are filled with the even-odd rule.
//!
//! Coordinates are millimetres in the lithophane plane: `x` to the right, `y` up
//! (it becomes the printer's Z axis). After [`Shape::normalized`] the bounding box
//! starts at the origin.

use std::collections::HashMap;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default)]
pub struct Shape {
    pub loops: Vec<Vec<DVec2>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisChoice {
    Auto,
    X,
    Y,
    Z,
}

impl Shape {
    pub fn rectangle(w: f64, h: f64, corner_radius: f64) -> Shape {
        let r = corner_radius.clamp(0.0, 0.5 * w.min(h));
        if r < 1e-3 {
            return Shape {
                loops: vec![vec![
                    DVec2::new(0.0, 0.0),
                    DVec2::new(w, 0.0),
                    DVec2::new(w, h),
                    DVec2::new(0.0, h),
                ]],
            };
        }
        let seg = ((r * 2.0).ceil() as usize).clamp(4, 48);
        let corners = [
            (DVec2::new(w - r, r), -90.0f64),
            (DVec2::new(w - r, h - r), 0.0),
            (DVec2::new(r, h - r), 90.0),
            (DVec2::new(r, r), 180.0),
        ];
        let mut lp = Vec::new();
        for (c, start) in corners {
            for k in 0..=seg {
                let a = (start + 90.0 * k as f64 / seg as f64).to_radians();
                lp.push(c + r * DVec2::new(a.cos(), a.sin()));
            }
        }
        Shape { loops: vec![lp] }.cleaned()
    }

    pub fn ellipse(w: f64, h: f64) -> Shape {
        let n = 256;
        let lp = (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / n as f64;
                DVec2::new(0.5 * w * (1.0 + a.cos()), 0.5 * h * (1.0 + a.sin()))
            })
            .collect();
        Shape { loops: vec![lp] }
    }

    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    pub fn bbox(&self) -> (DVec2, DVec2) {
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for p in self.loops.iter().flatten() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }

    pub fn size(&self) -> DVec2 {
        let (lo, hi) = self.bbox();
        hi - lo
    }

    #[cfg(test)]
    pub fn area(&self) -> f64 {
        crate::mesh::group_loops(&self.loops)
            .iter()
            .map(|(o, hs)| {
                crate::mesh::signed_area(&self.loops[*o]).abs()
                    - hs.iter()
                        .map(|&h| crate::mesh::signed_area(&self.loops[h]).abs())
                        .sum::<f64>()
            })
            .sum()
    }

    /// Moves the bounding box to start at the origin.
    pub fn normalized(mut self) -> Shape {
        let (lo, _) = self.bbox();
        for p in self.loops.iter_mut().flatten() {
            *p -= lo;
        }
        self
    }

    /// Rotates by `quarter_turns` × 90° counter-clockwise, optionally mirrors
    /// horizontally, scales, and normalizes.
    pub fn transformed(&self, quarter_turns: u8, mirror: bool, scale: f64) -> Shape {
        let mut s = self.clone();
        for p in s.loops.iter_mut().flatten() {
            let mut q = *p;
            for _ in 0..quarter_turns % 4 {
                q = DVec2::new(-q.y, q.x);
            }
            if mirror {
                q.x = -q.x;
            }
            *p = q * scale;
        }
        s.normalized()
    }

    /// Removes duplicate and collinear points and degenerate loops.
    pub fn cleaned(mut self) -> Shape {
        let (lo, hi) = self.bbox();
        let eps = 1e-7 * (hi - lo).max_element().max(1.0);
        for lp in &mut self.loops {
            lp.dedup_by(|a, b| a.distance(*b) < eps);
            while lp.len() > 1 && lp[0].distance(*lp.last().unwrap()) < eps {
                lp.pop();
            }
            // Drop points in the middle of straight runs.
            let collinear = |a: DVec2, b: DVec2, c: DVec2| {
                let cross = (b - a).perp_dot(c - b);
                let len = (b - a).length() * (c - b).length();
                cross.abs() <= 1e-12 * len.max(1e-300) && (b - a).dot(c - b) > 0.0
            };
            let mut out: Vec<DVec2> = Vec::with_capacity(lp.len());
            for &p in lp.iter() {
                while out.len() >= 2 && collinear(out[out.len() - 2], out[out.len() - 1], p) {
                    out.pop();
                }
                out.push(p);
            }
            while out.len() >= 3 && collinear(out[out.len() - 2], out[out.len() - 1], out[0]) {
                out.pop();
            }
            while out.len() >= 3 && collinear(out[out.len() - 1], out[0], out[1]) {
                out.remove(0);
            }
            *lp = out;
        }
        self.loops
            .retain(|lp| lp.len() >= 3 && crate::mesh::signed_area(lp).abs() > eps * eps);
        self
    }

    /// Sorted x coordinates where the horizontal line at `y` crosses the outline.
    pub fn crossings_h(&self, y: f64, out: &mut Vec<f64>) {
        out.clear();
        for lp in &self.loops {
            let n = lp.len();
            for i in 0..n {
                let a = lp[i];
                let b = lp[(i + 1) % n];
                if (a.y > y) != (b.y > y) {
                    out.push(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
                }
            }
        }
        out.sort_by(f64::total_cmp);
    }

    /// Sorted y coordinates where the vertical line at `x` crosses the outline.
    pub fn crossings_v(&self, x: f64, out: &mut Vec<f64>) {
        out.clear();
        for lp in &self.loops {
            let n = lp.len();
            for i in 0..n {
                let a = lp[i];
                let b = lp[(i + 1) % n];
                if (a.x > x) != (b.x > x) {
                    out.push(a.y + (x - a.x) * (b.y - a.y) / (b.x - a.x));
                }
            }
        }
        out.sort_by(f64::total_cmp);
    }

    pub fn contains(&self, p: DVec2) -> bool {
        self.loops
            .iter()
            .filter(|lp| crate::mesh::point_in_loop(p, lp))
            .count()
            % 2
            == 1
    }

    /// Lowest and highest outline point on the vertical line at `x`.
    pub fn vertical_extent(&self, x: f64) -> Option<(f64, f64)> {
        let mut c = Vec::new();
        self.crossings_v(x, &mut c);
        if c.len() >= 2 {
            Some((c[0], *c.last().unwrap()))
        } else {
            None
        }
    }

    pub fn edges(&self) -> impl Iterator<Item = (DVec2, DVec2)> + '_ {
        self.loops
            .iter()
            .flat_map(|lp| (0..lp.len()).map(move |i| (lp[i], lp[(i + 1) % lp.len()])))
    }
}

/// Answers "distance to the outline, if closer than `radius`" quickly.
pub struct EdgeIndex {
    origin: DVec2,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
    edges: Vec<(DVec2, DVec2)>,
    radius: f64,
}

impl EdgeIndex {
    pub fn new(shape: &Shape, radius: f64) -> EdgeIndex {
        let (lo, hi) = shape.bbox();
        let radius = radius.max(1e-6);
        let lo = lo - DVec2::splat(radius * 2.0);
        let hi = hi + DVec2::splat(radius * 2.0);
        let size = hi - lo;
        let mut cell = radius.max(0.5);
        while (size.x / cell) * (size.y / cell) > 250_000.0 {
            cell *= 1.5;
        }
        let nx = (size.x / cell).ceil() as usize + 1;
        let ny = (size.y / cell).ceil() as usize + 1;
        let mut buckets = vec![Vec::new(); nx * ny];
        let edges: Vec<_> = shape.edges().collect();
        for (k, (a, b)) in edges.iter().enumerate() {
            let elo = a.min(*b) - DVec2::splat(radius) - lo;
            let ehi = a.max(*b) + DVec2::splat(radius) - lo;
            let (i0, j0) = ((elo.x / cell) as usize, (elo.y / cell) as usize);
            let (i1, j1) = (
                ((ehi.x / cell) as usize).min(nx - 1),
                ((ehi.y / cell) as usize).min(ny - 1),
            );
            for j in j0..=j1 {
                for i in i0..=i1 {
                    buckets[j * nx + i].push(k as u32);
                }
            }
        }
        EdgeIndex {
            origin: lo,
            cell,
            nx,
            ny,
            buckets,
            edges,
            radius,
        }
    }

    /// Distance to the nearest outline edge, or `f64::INFINITY` if farther than the radius.
    pub fn distance(&self, p: DVec2) -> f64 {
        let q = (p - self.origin) / self.cell;
        if q.x < 0.0 || q.y < 0.0 {
            return f64::INFINITY;
        }
        let (i, j) = (q.x as usize, q.y as usize);
        if i >= self.nx || j >= self.ny {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        for &k in &self.buckets[j * self.nx + i] {
            let (a, b) = self.edges[k as usize];
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
            best = best.min(p.distance(a + ab * t));
        }
        if best <= self.radius {
            best
        } else {
            f64::INFINITY
        }
    }
}

/// Picks the axis along which an extruded polygon was extruded: every triangle of
/// such a mesh is either parallel (side walls) or perpendicular (caps) to it.
pub fn detect_extrusion_axis(tris: &[[[f32; 3]; 3]]) -> usize {
    let (lo, hi) = tri_bbox(tris);
    let extent = hi - lo;
    let mut score = [0.0f64; 3];
    let mut total = 0.0;
    for t in tris {
        let a = DVec3::from_array(t[0].map(|v| v as f64));
        let b = DVec3::from_array(t[1].map(|v| v as f64));
        let c = DVec3::from_array(t[2].map(|v| v as f64));
        let n = (b - a).cross(c - a);
        let area = n.length();
        if area <= 0.0 {
            continue;
        }
        let n = n / area;
        total += area;
        for (k, s) in score.iter_mut().enumerate() {
            let d = n[k].abs();
            if d > 0.999 || d < 0.001 {
                *s += area;
            }
        }
    }
    let candidates: Vec<usize> = (0..3)
        .filter(|&k| total > 0.0 && score[k] / total > 0.98)
        .collect();
    let pool = if candidates.is_empty() {
        vec![0, 1, 2]
    } else {
        candidates
    };
    // Among plausible axes, the extrusion is the thinnest direction.
    *pool
        .iter()
        .min_by(|&&a, &&b| extent[a].total_cmp(&extent[b]))
        .unwrap()
}

fn tri_bbox(tris: &[[[f32; 3]; 3]]) -> (DVec3, DVec3) {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for t in tris {
        for v in t {
            let p = DVec3::from_array(v.map(|x| x as f64));
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    (lo, hi)
}

/// 2D projection of a point for a given extrusion axis, as seen when looking at
/// the face of the extrusion (X: from +X, Y: from -Y, Z: from +Z).
fn project(p: DVec3, axis: usize) -> DVec2 {
    match axis {
        0 => DVec2::new(p.y, p.z),
        1 => DVec2::new(p.x, p.z),
        _ => DVec2::new(p.x, p.y),
    }
}

/// Cross-section of the mesh half-way along `axis`, as an even-odd shape.
pub fn slice_outline(tris: &[[[f32; 3]; 3]], axis: usize) -> Result<Shape, String> {
    if tris.is_empty() {
        return Err("the STL has no triangles".into());
    }
    let (lo, hi) = tri_bbox(tris);
    let extent = (hi - lo)[axis];
    if extent <= 0.0 {
        return Err("the STL is flat along the extrusion axis".into());
    }
    let level = lo[axis] + extent * (0.5 + 1.234e-5);

    // Weld the triangle soup so that cut points can be matched by mesh edge.
    let q = 1e-4;
    let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
    let mut verts: Vec<DVec3> = Vec::new();
    let mut weld = |v: &[f32; 3]| -> u32 {
        let key = v.map(|x| (x as f64 / q).round() as i64);
        *ids.entry(key).or_insert_with(|| {
            verts.push(DVec3::from_array(v.map(|x| x as f64)));
            (verts.len() - 1) as u32
        })
    };
    let welded: Vec<[u32; 3]> = tris
        .iter()
        .map(|t| [weld(&t[0]), weld(&t[1]), weld(&t[2])])
        .collect();

    let key = |a: u32, b: u32| -> u64 {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        ((a as u64) << 32) | b as u64
    };
    let mut points: HashMap<u64, DVec2> = HashMap::new();
    let mut segments: Vec<(u64, u64)> = Vec::new();
    for t in &welded {
        let mut cut = [0u64; 2];
        let mut n = 0;
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let (pa, pb) = (verts[a as usize], verts[b as usize]);
            if (pa[axis] > level) != (pb[axis] > level) {
                if n < 2 {
                    let id = key(a, b);
                    points.entry(id).or_insert_with(|| {
                        let (p0, p1) = if a < b { (pa, pb) } else { (pb, pa) };
                        let s = (level - p0[axis]) / (p1[axis] - p0[axis]);
                        project(p0 + (p1 - p0) * s, axis)
                    });
                    cut[n] = id;
                }
                n += 1;
            }
        }
        if n == 2 && cut[0] != cut[1] {
            segments.push((cut[0], cut[1]));
        }
    }

    let mut adjacency: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, s) in segments.iter().enumerate() {
        adjacency.entry(s.0).or_default().push(i);
        adjacency.entry(s.1).or_default().push(i);
    }
    let mut used = vec![false; segments.len()];
    let mut loops = Vec::new();
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let first = segments[start].0;
        let mut cur = segments[start].1;
        let mut lp = vec![points[&first]];
        while cur != first {
            lp.push(points[&cur]);
            let next = adjacency[&cur].iter().copied().find(|&s| !used[s]);
            let Some(s) = next else { break };
            used[s] = true;
            cur = if segments[s].0 == cur {
                segments[s].1
            } else {
                segments[s].0
            };
        }
        if lp.len() >= 3 {
            loops.push(lp);
        }
    }
    let shape = Shape { loops }.cleaned();
    if shape.is_empty() {
        return Err("could not find a closed outline in the STL".into());
    }
    Ok(shape)
}

/// Loads an STL and extracts the outline polygon of the extruded shape.
pub fn outline_from_triangles(
    tris: &[[[f32; 3]; 3]],
    axis: AxisChoice,
) -> Result<(Shape, usize), String> {
    let axis = match axis {
        AxisChoice::Auto => detect_extrusion_axis(tris),
        AxisChoice::X => 0,
        AxisChoice::Y => 1,
        AxisChoice::Z => 2,
    };
    Ok((slice_outline(tris, axis)?.normalized(), axis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::extrude;

    fn star_with_hole() -> Vec<Vec<DVec2>> {
        let star: Vec<DVec2> = (0..10)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 10.0;
                let r = if k % 2 == 0 { 40.0 } else { 18.0 };
                DVec2::new(r * a.cos(), r * a.sin())
            })
            .collect();
        let hole: Vec<DVec2> = (0..24)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 24.0;
                DVec2::new(5.0 * a.cos(), 5.0 * a.sin())
            })
            .collect();
        vec![star, hole]
    }

    fn soup(m: &crate::mesh::Mesh) -> Vec<[[f32; 3]; 3]> {
        m.triangles
            .iter()
            .map(|t| {
                [
                    m.positions[t[0] as usize],
                    m.positions[t[1] as usize],
                    m.positions[t[2] as usize],
                ]
            })
            .collect()
    }

    #[test]
    fn outline_from_extrusion_along_each_axis() {
        let loops = star_with_hole();
        let reference = Shape {
            loops: loops.clone(),
        }
        .normalized();
        for axis in 0..3 {
            let m = extrude(&loops, 0.0, 3.0, |p, w| match axis {
                0 => DVec3::new(w, p.x, p.y),
                1 => DVec3::new(p.x, w, p.y),
                _ => DVec3::new(p.x, p.y, w),
            });
            let tris = soup(&m);
            assert_eq!(detect_extrusion_axis(&tris), axis);
            let (shape, _) = outline_from_triangles(&tris, AxisChoice::Auto).unwrap();
            assert_eq!(shape.loops.len(), 2);
            assert!(
                (shape.area() - reference.area()).abs() < 1e-3,
                "axis {axis}"
            );
            assert!((shape.size() - reference.size()).length() < 1e-3);
        }
    }

    #[test]
    fn crossings_and_contains() {
        let s = Shape::rectangle(10.0, 20.0, 0.0);
        let mut c = Vec::new();
        s.crossings_h(5.0, &mut c);
        assert_eq!(c, vec![0.0, 10.0]);
        assert!(s.contains(DVec2::new(5.0, 5.0)));
        assert!(!s.contains(DVec2::new(15.0, 5.0)));
        assert_eq!(s.vertical_extent(3.0), Some((0.0, 20.0)));
    }

    #[test]
    fn edge_index_distance() {
        let s = Shape::rectangle(10.0, 10.0, 0.0);
        let idx = EdgeIndex::new(&s, 2.0);
        assert!((idx.distance(DVec2::new(1.0, 5.0)) - 1.0).abs() < 1e-9);
        assert!(idx.distance(DVec2::new(5.0, 5.0)).is_infinite());
    }
}
