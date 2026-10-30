//! Grayscale image handling and the brightness adjustments that drive thickness.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Loaded pictures are downscaled to this size: far finer than any printable relief, and it
/// keeps memory bounded for huge photos.
pub const MAX_IMAGE_DIM: usize = 4096;

/// Luminance image, values in 0..=1 (0 = black), rows top to bottom.
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Adjustments {
    /// Added to the result, -1..1.
    pub lightness: f32,
    /// -1..1, 0 = unchanged.
    pub contrast: f32,
    /// Midtone gamma, > 1 brightens.
    pub gamma: f32,
    /// Input level mapped to black, 0..1.
    pub black_point: f32,
    /// Input level mapped to white, 0..1.
    pub white_point: f32,
    pub invert: bool,
}

impl Default for Adjustments {
    fn default() -> Self {
        Adjustments {
            lightness: 0.0,
            contrast: 0.0,
            gamma: 1.0,
            black_point: 0.0,
            white_point: 1.0,
            invert: false,
        }
    }
}

impl Adjustments {
    pub fn apply(&self, v: f32) -> f32 {
        let mut v = if self.invert { 1.0 - v } else { v };
        let range = (self.white_point - self.black_point).max(1e-4);
        v = ((v - self.black_point) / range).clamp(0.0, 1.0);
        v = v.powf(1.0 / self.gamma.max(0.01));
        let c = self.contrast.clamp(-0.95, 0.95);
        v = (v - 0.5) * ((1.0 + c) / (1.0 - c)) + 0.5;
        (v + self.lightness).clamp(0.0, 1.0)
    }

    pub fn is_identity(&self) -> bool {
        *self == Adjustments::default()
    }
}

impl GrayImage {
    pub fn load(path: &Path) -> Result<GrayImage, String> {
        let img = image::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        Ok(Self::from_rgba(&img.to_rgba8()))
    }

    /// Transparent pixels count as white (they let the most light through).
    pub fn from_rgba(img: &image::RgbaImage) -> GrayImage {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let data = img
            .pixels()
            .map(|p| {
                let [r, g, b, a] = p.0.map(|c| c as f32 / 255.0);
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                lum * a + (1.0 - a)
            })
            .collect();
        GrayImage { w, h, data }
    }

    pub fn adjusted(&self, adj: &Adjustments) -> GrayImage {
        if adj.is_identity() {
            return self.clone();
        }
        // A lookup table keeps this fast for large images.
        const N: usize = 4096;
        let lut: Vec<f32> = (0..=N).map(|i| adj.apply(i as f32 / N as f32)).collect();
        let data = self
            .data
            .iter()
            .map(|&v| {
                let x = v.clamp(0.0, 1.0) * N as f32;
                let i = (x as usize).min(N - 1);
                let f = x - i as f32;
                lut[i] * (1.0 - f) + lut[i + 1] * f
            })
            .collect();
        GrayImage {
            w: self.w,
            h: self.h,
            data,
        }
    }

    /// Box-filter downscale so that neither side exceeds `max_dim`.
    pub fn downscaled(&self, max_dim: usize) -> GrayImage {
        let factor = self.w.max(self.h).div_ceil(max_dim.max(1));
        self.downscaled_by(factor)
    }

    pub fn downscaled_by(&self, factor: usize) -> GrayImage {
        if factor <= 1 {
            return self.clone();
        }
        let w = self.w.div_ceil(factor);
        let h = self.h.div_ceil(factor);
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut n = 0;
                for sy in y * factor..((y + 1) * factor).min(self.h) {
                    for sx in x * factor..((x + 1) * factor).min(self.w) {
                        sum += self.data[sy * self.w + sx];
                        n += 1;
                    }
                }
                data[y * w + x] = sum / n as f32;
            }
        }
        GrayImage { w, h, data }
    }

    /// Rotates 90° clockwise.
    pub fn rotated_cw(&self) -> GrayImage {
        let (w, h) = (self.h, self.w);
        let mut data = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = self.data[(self.h - 1 - x) * self.w + y];
            }
        }
        GrayImage { w, h, data }
    }

    /// Bilinear sample at pixel coordinates (pixel centres at +0.5).
    pub fn sample(&self, px: f64, py: f64) -> f32 {
        let x = (px - 0.5).clamp(0.0, (self.w - 1) as f64);
        let y = (py - 0.5).clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let at = |x: usize, y: usize| self.data[y * self.w + x];
        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    pub fn histogram(&self, bins: usize) -> Vec<u32> {
        let mut h = vec![0u32; bins];
        for &v in &self.data {
            h[((v.clamp(0.0, 1.0) * bins as f32) as usize).min(bins - 1)] += 1;
        }
        h
    }

    pub fn to_rgba(&self) -> Vec<u8> {
        self.data
            .iter()
            .flat_map(|&v| {
                let c = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                [c, c, c, 255]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjustments() {
        let id = Adjustments::default();
        assert!((id.apply(0.3) - 0.3).abs() < 1e-6);
        let levels = Adjustments {
            black_point: 0.2,
            white_point: 0.6,
            ..id
        };
        assert_eq!(levels.apply(0.1), 0.0);
        assert!((levels.apply(0.4) - 0.5).abs() < 1e-6);
        assert_eq!(levels.apply(0.9), 1.0);
        let inv = Adjustments { invert: true, ..id };
        assert!((inv.apply(0.25) - 0.75).abs() < 1e-6);
        let g = Adjustments { gamma: 2.0, ..id };
        assert!((g.apply(0.25) - 0.5).abs() < 1e-6);
        let c = Adjustments {
            contrast: 0.5,
            ..id
        };
        assert!(c.apply(0.6) > 0.6 && c.apply(0.4) < 0.4);
        let l = Adjustments {
            lightness: 0.1,
            ..id
        };
        assert!((l.apply(0.5) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn transforms() {
        let img = GrayImage {
            w: 3,
            h: 2,
            data: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        };
        let r = img.rotated_cw();
        assert_eq!((r.w, r.h), (2, 3));
        assert_eq!(r.data, vec![0.3, 0.0, 0.4, 0.1, 0.5, 0.2]);
        let d = img.downscaled_by(2);
        assert_eq!((d.w, d.h), (2, 1));
        assert!((d.data[0] - 0.2).abs() < 1e-6);
    }
}
