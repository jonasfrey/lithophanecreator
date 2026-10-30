//! Break-away supports that keep a vertically printed lithophane from wobbling.
//!
//! Reverse-engineered from the example print: at each side of the plate stand two
//! thin triangular buttress walls (one on each face), perpendicular to the plate
//! and almost touching it along the full height. Each wall stands on a 0.4 mm
//! foot plate for bed adhesion and is stiffened by a rib and a 60° fillet.
//! Here the two foot plates of a side are joined around the end of the plate,
//! so every side is one support part.
//!
//! For outlines without a flat bottom, a "cradle" fills the space between the bed
//! and the lower edge of the outline (with a small gap, like slicer support), and
//! is split into a left and a right half that belong to the two support parts.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::mesh::{Mesh, extrude};
use crate::shape::Shape;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SupportParams {
    pub enabled: bool,
    /// Place the walls automatically where the outline is tall enough.
    pub auto_position: bool,
    /// Distance of the walls from the left/right end of the outline (manual mode).
    pub inset: f64,
    pub wall_thickness: f64,
    /// Wall depth at the bed as a fraction of its height.
    pub depth_ratio: f64,
    pub min_depth: f64,
    /// Wall depth at the top.
    pub top_width: f64,
    pub foot_thickness: f64,
    /// How far the foot plate reaches towards the middle of the plate.
    pub foot_width: f64,
    /// How far the foot plate reaches beyond the wall's base.
    pub foot_extra: f64,
    /// Clearance between foot plate and lithophane.
    pub foot_gap: f64,
    /// Clearance between the walls and the lithophane faces.
    pub contact_gap: f64,
    pub ribs: bool,
    pub cradle: bool,
    /// Vertical clearance between cradle and lithophane.
    pub cradle_gap: f64,
}

impl Default for SupportParams {
    fn default() -> Self {
        // Proportions of the example: 144 mm tall plate, walls 28.5 mm deep at the
        // bed and 7.3 mm at the top, 0.8 mm thick, on 0.4 mm feet.
        SupportParams {
            enabled: true,
            auto_position: true,
            inset: 0.0,
            wall_thickness: 0.8,
            depth_ratio: 0.2,
            min_depth: 10.0,
            top_width: 7.3,
            foot_thickness: 0.4,
            foot_width: 45.0,
            foot_extra: 12.0,
            foot_gap: 1.4,
            contact_gap: 0.01,
            ribs: true,
            cradle: true,
            cradle_gap: 0.05,
        }
    }
}

/// Where a support wall pair touches the lithophane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SidePlan {
    pub x0: f64,
    pub x1: f64,
    /// Height of the contact (top of the outline above the wall).
    pub height: f64,
    /// +1 for the left support (its inside points to +x), -1 for the right one.
    pub sign: f64,
}

fn extent_over(shape: &Shape, x0: f64, x1: f64) -> Option<(f64, f64)> {
    let eps = (x1 - x0) * 0.02;
    let mut bottom = f64::NEG_INFINITY;
    let mut top = f64::INFINITY;
    for x in [x0 + eps, 0.5 * (x0 + x1), x1 - eps] {
        let (b, t) = shape.vertical_extent(x)?;
        bottom = bottom.max(b);
        top = top.min(t);
    }
    (top > bottom).then_some((bottom, top))
}

pub fn plan(shape: &Shape, p: &SupportParams) -> Option<[SidePlan; 2]> {
    if !p.enabled || shape.is_empty() {
        return None;
    }
    let size = shape.size();
    let wt = p.wall_thickness.clamp(0.2, size.x / 6.0);
    let max_inset = (size.x * 0.5 - wt - 0.5).max(0.0);
    let wall = |inset: f64, sign: f64| {
        if sign > 0.0 {
            (inset, inset + wt)
        } else {
            (size.x - inset - wt, size.x - inset)
        }
    };
    let pick = |sign: f64| -> Option<SidePlan> {
        let inset = if p.auto_position {
            // Walk inwards until the outline is at least 75% of its full height.
            let mut best = (0.0, f64::NEG_INFINITY);
            let mut chosen = None;
            let mut d = 0.0;
            while d <= max_inset.min(size.x / 3.0) {
                let (x0, x1) = wall(d, sign);
                if let Some((_, top)) = extent_over(shape, x0, x1) {
                    if top >= 0.75 * size.y {
                        chosen = Some(d);
                        break;
                    }
                    if top > best.1 {
                        best = (d, top);
                    }
                }
                d += 0.25;
            }
            chosen.unwrap_or(best.0)
        } else {
            p.inset.clamp(0.0, max_inset)
        };
        let (x0, x1) = wall(inset, sign);
        let (_, top) = extent_over(shape, x0, x1)?;
        Some(SidePlan {
            x0,
            x1,
            height: top,
            sign,
        })
    };
    Some([pick(1.0)?, pick(-1.0)?])
}

/// X-ranges of the lithophane that are made full thickness so the walls touch a flat face.
pub fn contact_strips(plan: &Option<[SidePlan; 2]>) -> Vec<(f64, f64)> {
    plan.iter()
        .flatten()
        .map(|s| (s.x0 - 0.3, s.x1 + 0.3))
        .collect()
}

/// Region under the outline's lower edge, split at the middle into left and right pieces.
fn cradle_loops(shape: &Shape, gap: f64) -> [Vec<Vec<DVec2>>; 2] {
    let size = shape.size();
    // Profile points divide the width evenly, so a mirrored outline gets a mirrored cradle.
    let n = (size.x / 0.05f64.min(size.x / 400.0).max(0.01)).ceil() as usize;
    let step = size.x / n as f64;
    let split = size.x * 0.5;
    let min_h = 0.4;
    let bottom = |x: f64| {
        shape
            .vertical_extent(x.clamp(1e-6, size.x - 1e-6))
            .map(|e| e.0)
    };
    // The outline's lowest point over an x-range lies at one of the range's ends or at an
    // outline vertex inside it (vertices never lie below the lower edge).
    let mut vertices: Vec<DVec2> = shape.loops.iter().flatten().copied().collect();
    vertices.sort_by(|a, b| a.x.total_cmp(&b.x));
    let lowest = |x0: f64, x1: f64| -> Option<f64> {
        let start = vertices.partition_point(|v| v.x < x0);
        let inside = vertices[start..].iter().take_while(|v| v.x <= x1);
        [bottom(x0), bottom(x1)]
            .into_iter()
            .flatten()
            .chain(inside.map(|v| v.y))
            .reduce(f64::min)
    };
    // Each profile point keeps the gap to the outline within one step (plus the gap) on
    // either side, so the straight segments between points keep it too, also sideways.
    let reach = step + gap;
    let tops: Vec<Option<f64>> = (0..=n)
        .map(|k| {
            let x = k as f64 * step;
            lowest(x - reach, x + reach)
                .map(|b| b - gap)
                .filter(|&t| t >= min_h)
        })
        .collect();
    let mut out: [Vec<Vec<DVec2>>; 2] = [Vec::new(), Vec::new()];
    let mut run: Vec<DVec2> = Vec::new();
    let mut flush = |run: &mut Vec<DVec2>, side: usize| {
        if run.len() >= 2 && run.last().unwrap().x - run[0].x >= 0.5 {
            let (xa, xb) = (run[0].x, run.last().unwrap().x);
            let mut lp = vec![DVec2::new(xa, 0.0), DVec2::new(xb, 0.0)];
            lp.extend(run.iter().rev());
            out[side].push(lp);
        }
        run.clear();
    };
    let mut run_side = 0;
    for k in 0..n {
        let (xa, xb) = (k as f64 * step, (k + 1) as f64 * step);
        let side = if xa + xb < size.x { 0 } else { 1 };
        // The two halves stay apart around the middle.
        let usable =
            (xb <= split - 0.2 || xa >= split + 0.2) && tops[k].is_some() && tops[k + 1].is_some();
        if !usable || side != run_side {
            flush(&mut run, run_side);
            run_side = side;
        }
        if usable {
            if run.is_empty() {
                run.push(DVec2::new(xa, tops[k].unwrap()));
            }
            run.push(DVec2::new(xb, tops[k + 1].unwrap()));
        }
    }
    flush(&mut run, run_side);
    out
}

/// Builds the left and right support parts.
pub fn build(
    shape: &Shape,
    plan: &[SidePlan; 2],
    max_thickness: f64,
    p: &SupportParams,
) -> [Mesh; 2] {
    let size = shape.size();
    let tm = max_thickness;
    let cg = p.contact_gap.max(0.0);
    let cradles = if p.cradle {
        cradle_loops(shape, p.cradle_gap.max(0.05))
    } else {
        Default::default()
    };
    let mut parts: [Mesh; 2] = [Mesh::default(), Mesh::default()];
    for (side, s) in plan.iter().enumerate() {
        let part = &mut parts[side];
        let h = s.height;
        let depth = (p.depth_ratio * h).max(p.min_depth);
        let top_w = p.top_width.clamp(1.0, 0.5 * depth);
        let h_low = (0.056 * h).clamp(1.0, 8.0).min(0.5 * h);
        let (x_outer, x_inner) = if s.sign > 0.0 {
            (s.x0, s.x1)
        } else {
            (s.x1, s.x0)
        };
        // Distance `d` from the plate on the back (+Y) and front (-Y) face.
        let faces: [Box<dyn Fn(f64) -> f64>; 2] =
            [Box::new(move |d| cg + d), Box::new(move |d| -tm - cg - d)];

        let wall = vec![vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(depth, 0.0),
            DVec2::new(depth, h_low),
            DVec2::new(top_w, h),
            DVec2::new(0.0, h),
        ]];
        for y in &faces {
            part.append(&extrude(&wall, s.x0, s.x1, |q, w| {
                DVec3::new(w, y(q.x), q.y)
            }));
        }

        if p.ribs {
            // Rib on the inner side of each wall, near the plate where the wall is narrow.
            let rib_d0 = (0.75 * top_w - 0.4).max(0.5);
            let rib_d1 = (rib_d0 + 0.8).min(top_w);
            let rib = vec![vec![
                DVec2::new(-0.02, 0.0),
                DVec2::new(2.0, 0.0),
                DVec2::new(2.0, 0.7 * h),
                DVec2::new(0.25, h),
                DVec2::new(-0.02, h),
            ]];
            let fh = (0.95 * h_low).min(8.26);
            let fillet = vec![vec![
                DVec2::new(-0.02, 0.0),
                DVec2::new(fh * 0.577, 0.0),
                DVec2::new(-0.02, fh),
            ]];
            for y in &faces {
                let place = |q: DVec2, w: f64| DVec3::new(x_inner + s.sign * q.x, y(w), q.y);
                part.append(&extrude(&rib, rib_d0, rib_d1, place));
                part.append(&extrude(&fillet, 0.4, (0.55 * depth).max(1.0), place));
            }
        }

        // Foot plate: one "C" shape around the end of the plate, joining both faces.
        let bridge = 3.0;
        let foot_len = depth + p.foot_extra;
        let reach = p
            .foot_width
            .min(size.x * 0.5 - 0.5 - (x_outer - if s.sign > 0.0 { 0.0 } else { size.x }).abs())
            .max(p.wall_thickness + 4.0);
        let (xo, xb) = if s.sign > 0.0 {
            (-p.foot_gap - bridge, -p.foot_gap)
        } else {
            (size.x + p.foot_gap + bridge, size.x + p.foot_gap)
        };
        let xi = x_outer + s.sign * reach;
        let (yf_out, yf_in) = (-tm - cg - foot_len, -tm - p.foot_gap);
        let (yb_in, yb_out) = (p.foot_gap, cg + foot_len);
        let foot = vec![vec![
            DVec2::new(xo, yf_out),
            DVec2::new(xi, yf_out),
            DVec2::new(xi, yf_in),
            DVec2::new(xb, yf_in),
            DVec2::new(xb, yb_in),
            DVec2::new(xi, yb_in),
            DVec2::new(xi, yb_out),
            DVec2::new(xo, yb_out),
        ]];
        part.append(&extrude(&foot, 0.0, p.foot_thickness.max(0.1), |q, w| {
            DVec3::new(q.x, q.y, w)
        }));

        // Cradle pieces sit inside the plate's slab and reach slightly into the walls.
        if !cradles[side].is_empty() {
            let (y0, y1) = (-tm - cg - 0.05, cg + 0.05);
            part.append(&extrude(&cradles[side], y0, y1, |q, w| {
                DVec3::new(q.x, w, q.y)
            }));
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_supports_match_example() {
        let shape = Shape::rectangle(108.0, 144.0, 0.0);
        let p = SupportParams::default();
        let plan = plan(&shape, &p).unwrap();
        assert!(plan[0].x0.abs() < 1e-9 && (plan[0].x1 - 0.8).abs() < 1e-9);
        assert!((plan[1].x1 - 108.0).abs() < 1e-9);
        assert!((plan[0].height - 144.0).abs() < 1e-9);
        let parts = build(&shape, &plan, 3.2, &p);
        for part in &parts {
            let (lo, hi) = part.bbox().unwrap();
            assert!(lo.z.abs() < 1e-6 && (hi.z - 144.0).abs() < 1e-4);
            // Wall depth ~ 28.8 mm plus foot extension on both faces.
            assert!(hi.y > 40.0 && lo.y < -43.0);
            assert!(part.signed_volume() > 0.0);
        }
        // No cradle for a flat bottom.
        assert!(cradle_loops(&shape, 0.2).iter().all(|c| c.is_empty()));
    }

    #[test]
    fn ellipse_gets_cradle_and_inset_walls() {
        let shape = Shape::ellipse(100.0, 80.0);
        let p = SupportParams::default();
        let plan = plan(&shape, &p).unwrap();
        assert!(plan[0].x0 > 1.0 && plan[0].x0 < 20.0);
        assert!(plan[0].height >= 0.75 * 80.0);
        let cradles = cradle_loops(&shape, p.cradle_gap);
        assert!(!cradles[0].is_empty() && !cradles[1].is_empty());
        // The cradle keeps its gap below the outline, also between its profile points.
        for lp in cradles.iter().flatten() {
            for (a, b) in lp[2..].iter().zip(&lp[3..]) {
                for k in 0..=20 {
                    let q = a.lerp(*b, k as f64 / 20.0);
                    let (bottom, _) = shape.vertical_extent(q.x.clamp(0.01, 99.99)).unwrap();
                    assert!(
                        q.y <= bottom - p.cradle_gap + 1e-6,
                        "cradle point {q:?} vs bottom {bottom}"
                    );
                }
            }
        }
        let parts = build(&shape, &plan, 3.2, &p);
        assert!(parts.iter().all(|m| !m.triangles.is_empty()));
    }

    /// Sorted, rounded vertex positions, optionally mirrored at x = w / 2.
    fn vertex_set(m: &Mesh, mirror_width: Option<f64>) -> Vec<[i64; 3]> {
        let mut v: Vec<[i64; 3]> = m
            .positions
            .iter()
            .map(|p| {
                let x = match mirror_width {
                    Some(w) => w - p[0] as f64,
                    None => p[0] as f64,
                };
                [x, p[1] as f64, p[2] as f64].map(|c| (c * 1e3).round() as i64)
            })
            .collect();
        v.sort();
        v.dedup();
        v
    }

    #[test]
    fn symmetric_outlines_get_mirrored_supports() {
        // Widths that the cradle's sample spacing doesn't divide evenly.
        for shape in [
            Shape::ellipse(100.0, 80.0),
            Shape::ellipse(83.7, 61.3),
            Shape::rectangle(71.3, 90.0, 20.0),
        ] {
            let w = shape.size().x;
            let p = SupportParams::default();
            let plan = plan(&shape, &p).unwrap();
            assert!((plan[0].x0 - (w - plan[1].x1)).abs() < 1e-9);
            let [left, right] = build(&shape, &plan, 3.2, &p);
            assert_eq!(vertex_set(&left, None), vertex_set(&right, Some(w)));
        }
    }
}
