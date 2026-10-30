//! Triangle meshes, polygon extrusion and binary STL output.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use glam::{DVec2, DVec3};

/// An indexed triangle mesh. Triangles are counter-clockwise when seen from outside.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    /// Optional per-vertex scalar (lithophane thickness, or -1 for the flat back).
    pub attr: Vec<f32>,
}

impl Mesh {
    pub fn add_vertex(&mut self, p: DVec3) -> u32 {
        self.positions.push([p.x as f32, p.y as f32, p.z as f32]);
        (self.positions.len() - 1) as u32
    }

    pub fn append(&mut self, other: &Mesh) {
        let offset = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.triangles.extend(
            other
                .triangles
                .iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]),
        );
        if !self.attr.is_empty() || !other.attr.is_empty() {
            self.attr.resize(offset as usize, 0.0);
            if other.attr.is_empty() {
                self.attr.resize(self.positions.len(), 0.0);
            } else {
                self.attr.extend_from_slice(&other.attr);
            }
        }
    }

    pub fn translate(&mut self, d: DVec3) {
        for p in &mut self.positions {
            p[0] = (p[0] as f64 + d.x) as f32;
            p[1] = (p[1] as f64 + d.y) as f32;
            p[2] = (p[2] as f64 + d.z) as f32;
        }
    }

    pub fn bbox(&self) -> Option<(DVec3, DVec3)> {
        let mut it = self.positions.iter().map(|p| to_dvec3(*p));
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p))))
    }

    /// Signed volume; positive for a closed mesh with outward-facing triangles.
    pub fn signed_volume(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let a = to_dvec3(self.positions[t[0] as usize]);
                let b = to_dvec3(self.positions[t[1] as usize]);
                let c = to_dvec3(self.positions[t[2] as usize]);
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    pub fn flip(&mut self) {
        for t in &mut self.triangles {
            t.swap(1, 2);
        }
    }

    /// Makes a closed shell face outwards.
    pub fn orient_outward(&mut self) {
        if self.signed_volume() < 0.0 {
            self.flip();
        }
    }

    pub fn triangle_normal(&self, t: &[u32; 3]) -> DVec3 {
        let a = to_dvec3(self.positions[t[0] as usize]);
        let b = to_dvec3(self.positions[t[1] as usize]);
        let c = to_dvec3(self.positions[t[2] as usize]);
        (b - a).cross(c - a).normalize_or_zero()
    }

    /// Number of edges not shared by exactly one opposite-oriented triangle.
    /// Zero means the mesh is a closed, consistently oriented 2-manifold.
    pub fn open_edge_count(&self) -> usize {
        let mut edges: HashMap<(u32, u32), i32> = HashMap::with_capacity(self.triangles.len() * 3);
        for t in &self.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edges.entry((a, b)).or_default() += 1;
            }
        }
        edges
            .iter()
            .filter(|&(&(a, b), &n)| n != 1 || edges.get(&(b, a)).copied() != Some(1))
            .count()
    }
}

pub fn to_dvec3(p: [f32; 3]) -> DVec3 {
    DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

pub fn signed_area(lp: &[DVec2]) -> f64 {
    let n = lp.len();
    (0..n).map(|i| lp[i].perp_dot(lp[(i + 1) % n])).sum::<f64>() * 0.5
}

pub fn point_in_loop(p: DVec2, lp: &[DVec2]) -> bool {
    let mut inside = false;
    let n = lp.len();
    for i in 0..n {
        let a = lp[i];
        let b = lp[(i + 1) % n];
        if (a.y > p.y) != (b.y > p.y) {
            let x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if x < p.x {
                inside = !inside;
            }
        }
    }
    inside
}

/// Groups even-odd loops into (outer, holes) sets and orients outers CCW and holes CW.
/// Returns indices into `loops`.
pub fn group_loops(loops: &[Vec<DVec2>]) -> Vec<(usize, Vec<usize>)> {
    // Nesting depth decides the role: even depth = outer, odd depth = hole.
    let depth: Vec<usize> = (0..loops.len())
        .map(|i| {
            let p = loops[i][0];
            (0..loops.len())
                .filter(|&j| j != i && point_in_loop(p, &loops[j]))
                .count()
        })
        .collect();
    let area: Vec<f64> = loops.iter().map(|l| signed_area(l).abs()).collect();
    let mut groups: Vec<(usize, Vec<usize>)> = (0..loops.len())
        .filter(|&i| depth[i].is_multiple_of(2))
        .map(|i| (i, Vec::new()))
        .collect();
    for h in (0..loops.len()).filter(|&i| depth[i] % 2 == 1) {
        // The hole belongs to the smallest outer (with depth one less) that contains it.
        let p = loops[h][0];
        let owner = groups
            .iter_mut()
            .filter(|(o, _)| depth[*o] + 1 == depth[h] && point_in_loop(p, &loops[*o]))
            .min_by(|a, b| area[a.0].total_cmp(&area[b.0]));
        if let Some((_, holes)) = owner {
            holes.push(h);
        }
    }
    groups
}

/// Triangulates a polygon given as point references into `pts`.
/// `outer` must be CCW, `holes` CW. The returned triangles are CCW and use every
/// boundary point (earcut drops collinear points; those edges are re-split here so
/// that the result has no T-junctions with neighbouring geometry).
/// Extra (Steiner) points are returned separately; their indices start at `pts.len()`.
/// Returns `None` if the triangulation does not cover the polygon exactly.
pub fn triangulate(
    pts: &[DVec2],
    outer: &[u32],
    holes: &[Vec<u32>],
) -> Option<(Vec<[u32; 3]>, Vec<DVec2>)> {
    let mut tris = earcut_raw(pts, outer, holes);

    // Re-insert boundary points that earcut skipped.
    let mut used = vec![false; pts.len()];
    for t in &tris {
        for &i in t {
            used[i as usize] = true;
        }
    }
    let mut skipped: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut extra_pts: Vec<DVec2> = Vec::new();
    for lp in std::iter::once(outer).chain(holes.iter().map(|h| h.as_slice())) {
        let n = lp.len();
        let start = (0..n).find(|&i| used[lp[i] as usize])?;
        let mut k = start;
        loop {
            let a = lp[k];
            let mut between = Vec::new();
            let mut j = (k + 1) % n;
            while !used[lp[j] as usize] {
                between.push(lp[j]);
                j = (j + 1) % n;
            }
            if !between.is_empty() {
                skipped.insert((a, lp[j]), between);
            }
            k = j;
            if k == start {
                break;
            }
        }
    }
    if !skipped.is_empty() {
        let mut out = Vec::with_capacity(tris.len() + skipped.len() * 4);
        for t in tris {
            let mut ring: Vec<u32> = Vec::new();
            let mut split = false;
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                ring.push(a);
                if let Some(between) = skipped.get(&(a, b)) {
                    ring.extend_from_slice(between);
                    split = true;
                } else if let Some(between) = skipped.get(&(b, a)) {
                    ring.extend(between.iter().rev());
                    split = true;
                }
            }
            if !split {
                out.push(t);
                continue;
            }
            // Fan around the triangle centroid (a new Steiner point).
            let c = (pts[t[0] as usize] + pts[t[1] as usize] + pts[t[2] as usize]) / 3.0;
            let ci = (pts.len() + extra_pts.len()) as u32;
            extra_pts.push(c);
            for k in 0..ring.len() {
                out.push([ring[k], ring[(k + 1) % ring.len()], ci]);
            }
        }
        tris = out;
    }

    // Validate: area must match and every boundary edge must be used exactly once.
    let all_pts = |i: u32| -> DVec2 {
        if (i as usize) < pts.len() {
            pts[i as usize]
        } else {
            extra_pts[i as usize - pts.len()]
        }
    };
    let tri_area: f64 = tris
        .iter()
        .map(|t| (all_pts(t[1]) - all_pts(t[0])).perp_dot(all_pts(t[2]) - all_pts(t[0])) * 0.5)
        .sum();
    let loop_area = |lp: &[u32]| -> f64 {
        let n = lp.len();
        (0..n)
            .map(|i| pts[lp[i] as usize].perp_dot(pts[lp[(i + 1) % n] as usize]))
            .sum::<f64>()
            * 0.5
    };
    let poly_area = loop_area(outer) + holes.iter().map(|h| loop_area(h)).sum::<f64>();
    if (tri_area - poly_area).abs() > 1e-6 * poly_area.abs().max(1e-9) + 1e-9 {
        return None;
    }
    let mut edge_use: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &tris {
        for k in 0..3 {
            *edge_use.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    for lp in std::iter::once(outer).chain(holes.iter().map(|h| h.as_slice())) {
        for i in 0..lp.len() {
            if edge_use.get(&(lp[i], lp[(i + 1) % lp.len()])).copied() != Some(1) {
                return None;
            }
        }
    }
    Some((tris, extra_pts))
}

/// Plain earcut triangulation (CCW triangles); may skip collinear boundary points.
pub fn earcut_raw(pts: &[DVec2], outer: &[u32], holes: &[Vec<u32>]) -> Vec<[u32; 3]> {
    let mut flat: Vec<u32> = outer.to_vec();
    let mut hole_idx: Vec<u32> = Vec::new();
    for h in holes {
        hole_idx.push(flat.len() as u32);
        flat.extend_from_slice(h);
    }
    let mut local_tris: Vec<u32> = Vec::new();
    earcut::Earcut::<f64>::new().earcut(
        flat.iter().map(|&i| [pts[i as usize].x, pts[i as usize].y]),
        &hole_idx,
        &mut local_tris,
    );
    local_tris
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| {
            let t = [
                flat[c[0] as usize],
                flat[c[1] as usize],
                flat[c[2] as usize],
            ];
            let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
            if (b - a).perp_dot(c - a) < 0.0 {
                [t[0], t[2], t[1]]
            } else {
                t
            }
        })
        .collect()
}

/// Extrudes even-odd 2D loops between `w0` and `w1`. `map(p, w)` places a 2D point at
/// extrusion coordinate `w` in 3D. The result is a closed, outward-facing shell.
pub fn extrude(loops: &[Vec<DVec2>], w0: f64, w1: f64, map: impl Fn(DVec2, f64) -> DVec3) -> Mesh {
    let mut mesh = Mesh::default();
    let mut pts: Vec<DVec2> = Vec::new();
    let mut oriented: Vec<Vec<u32>> = Vec::new();
    let groups = group_loops(loops);
    let mut group_refs: Vec<(usize, Vec<usize>)> = Vec::new();
    for (outer, holes) in &groups {
        let mut add = |li: usize, ccw: bool| -> usize {
            let mut lp = loops[li].clone();
            if (signed_area(&lp) > 0.0) != ccw {
                lp.reverse();
            }
            let base = pts.len() as u32;
            pts.extend_from_slice(&lp);
            oriented.push((0..lp.len() as u32).map(|i| base + i).collect());
            oriented.len() - 1
        };
        let o = add(*outer, true);
        let hs = holes.iter().map(|&h| add(h, false)).collect();
        group_refs.push((o, hs));
    }

    let mut all_pts = pts.clone();
    let mut cap: Vec<[u32; 3]> = Vec::new();
    for (o, hs) in &group_refs {
        let holes: Vec<Vec<u32>> = hs.iter().map(|&h| oriented[h].clone()).collect();
        let (tris, steiner) = triangulate(&all_pts, &oriented[*o], &holes)
            .unwrap_or_else(|| (earcut_raw(&all_pts, &oriented[*o], &holes), Vec::new()));
        // Steiner indices are relative to all_pts.len() at call time.
        all_pts.extend(steiner);
        cap.extend(tris);
    }
    let m = all_pts.len() as u32;
    for &p in &all_pts {
        mesh.add_vertex(map(p, w0));
    }
    for &p in &all_pts {
        mesh.add_vertex(map(p, w1));
    }
    for t in &cap {
        mesh.triangles.push([t[0] + m, t[1] + m, t[2] + m]);
        mesh.triangles.push([t[0], t[2], t[1]]);
    }
    for lp in &oriented {
        for i in 0..lp.len() {
            let a = lp[i];
            let b = lp[(i + 1) % lp.len()];
            mesh.triangles.push([a, b, b + m]);
            mesh.triangles.push([a, b + m, a + m]);
        }
    }
    mesh.orient_outward();
    mesh
}

/// Writes all meshes into one binary STL file.
pub fn write_stl(path: &Path, meshes: &[&Mesh]) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut header = [0u8; 80];
    let text = b"binary STL - lithophane-creator";
    header[..text.len()].copy_from_slice(text);
    w.write_all(&header)?;
    let count: usize = meshes.iter().map(|m| m.triangles.len()).sum();
    w.write_all(&(count as u32).to_le_bytes())?;
    let mut buf = [0u8; 50];
    for m in meshes {
        for t in &m.triangles {
            let n = m.triangle_normal(t);
            let vals = [
                n.x as f32,
                n.y as f32,
                n.z as f32,
                m.positions[t[0] as usize][0],
                m.positions[t[0] as usize][1],
                m.positions[t[0] as usize][2],
                m.positions[t[1] as usize][0],
                m.positions[t[1] as usize][1],
                m.positions[t[1] as usize][2],
                m.positions[t[2] as usize][0],
                m.positions[t[2] as usize][1],
                m.positions[t[2] as usize][2],
            ];
            for (k, v) in vals.iter().enumerate() {
                buf[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            w.write_all(&buf)?;
        }
    }
    w.flush()
}

/// Reads an ASCII or binary STL into a triangle soup.
pub fn read_stl(path: &Path) -> Result<Vec<[[f32; 3]; 3]>, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_stl(&data).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse_stl(data: &[u8]) -> Result<Vec<[[f32; 3]; 3]>, String> {
    if data.len() >= 84 {
        let n = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
        if 84 + n * 50 == data.len() {
            let mut tris = Vec::with_capacity(n);
            for i in 0..n {
                let rec = &data[84 + i * 50..84 + i * 50 + 50];
                let f = |k: usize| f32::from_le_bytes(rec[k * 4..k * 4 + 4].try_into().unwrap());
                tris.push([[f(3), f(4), f(5)], [f(6), f(7), f(8)], [f(9), f(10), f(11)]]);
            }
            return Ok(tris);
        }
    }
    let text = std::str::from_utf8(data).map_err(|_| "not a valid STL file".to_string())?;
    let mut tris = Vec::new();
    let mut cur: Vec<[f32; 3]> = Vec::with_capacity(3);
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() == Some("vertex") {
            let v: Vec<f32> = it.filter_map(|s| s.parse().ok()).collect();
            if v.len() != 3 {
                return Err(format!("malformed vertex line: {line}"));
            }
            cur.push([v[0], v[1], v[2]]);
            if cur.len() == 3 {
                tris.push([cur[0], cur[1], cur[2]]);
                cur.clear();
            }
        }
    }
    if tris.is_empty() {
        return Err("STL file contains no triangles".into());
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_with_hole() -> Vec<Vec<DVec2>> {
        let sq = |c: f64, r: f64| {
            vec![
                DVec2::new(c - r, c - r),
                DVec2::new(c + r, c - r),
                DVec2::new(c + r, c + r),
                DVec2::new(c - r, c + r),
            ]
        };
        vec![sq(0.0, 10.0), sq(0.0, 3.0)]
    }

    #[test]
    fn extrusion_is_closed() {
        let m = extrude(&square_with_hole(), 0.0, 2.0, |p, w| {
            DVec3::new(p.x, p.y, w)
        });
        assert_eq!(m.open_edge_count(), 0);
        let expected = (400.0 - 36.0) * 2.0;
        assert!((m.signed_volume() - expected).abs() < 1e-6);
    }

    #[test]
    fn collinear_points_are_kept() {
        // A square whose edges carry many collinear points (like marching-squares output).
        let mut lp = Vec::new();
        for i in 0..10 {
            lp.push(DVec2::new(i as f64, 0.0));
        }
        for i in 0..10 {
            lp.push(DVec2::new(10.0, i as f64));
        }
        for i in 0..10 {
            lp.push(DVec2::new(10.0 - i as f64, 10.0));
        }
        for i in 0..10 {
            lp.push(DVec2::new(0.0, 10.0 - i as f64));
        }
        let m = extrude(&[lp], 0.0, 1.0, |p, w| DVec3::new(p.x, w, p.y));
        assert_eq!(m.open_edge_count(), 0);
        assert!((m.signed_volume() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn stl_roundtrip() {
        let m = extrude(&square_with_hole(), 0.0, 2.0, |p, w| {
            DVec3::new(p.x, p.y, w)
        });
        let dir = std::env::temp_dir().join(format!("litho-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.stl");
        write_stl(&path, &[&m]).unwrap();
        let tris = read_stl(&path).unwrap();
        assert_eq!(tris.len(), m.triangles.len());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ascii_stl() {
        let s = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let tris = parse_stl(s.as_bytes()).unwrap();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0][1], [1.0, 0.0, 0.0]);
    }
}
