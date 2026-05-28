use crate::config::ColorScheme;
use crate::zoom::{ZoomState, ZoomAnimation};
use crate::renderer::{Renderer, generate_color_lut};
use std::simd::f64x4;
use std::simd::u32x4;
use std::simd::Mask;
use std::simd::Select;
use std::simd::cmp::SimdPartialOrd;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

/// 标量 Mandelbrot 计算（用于残余像素）
#[inline(always)]
fn mandelbrot(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let q = (c_re - 0.25).powi(2) + c_im.powi(2);
    if q * (q + (c_re - 0.25)) <= 0.25 * c_im.powi(2) {
        return max_iter;
    }
    if (c_re + 1.0).powi(2) + c_im.powi(2) <= 0.0625 {
        return max_iter;
    }

    let mut z_re = 0.0;
    let mut z_im = 0.0;
    let mut z_re_old = 0.0;
    let mut z_im_old = 0.0;
    let mut period = 0;

    for i in 0..max_iter {
        let z_re_squared = z_re * z_re;
        let z_im_squared = z_im * z_im;

        if z_re_squared + z_im_squared > 4.0 {
            return i;
        }

        let z_im_new = 2.0 * z_re * z_im + c_im;
        z_re = z_re_squared - z_im_squared + c_re;
        z_im = z_im_new;

        if z_re == z_re_old && z_im == z_im_old {
            return max_iter;
        }

        period += 1;
        if period > 20 {
            period = 0;
            z_re_old = z_re;
            z_im_old = z_im;
        }
    }

    max_iter
}

/// SIMD Mandelbrot 计算（f64x4）
#[inline(always)]
fn mandelbrot_simd_f64x4(
    c_re: f64x4,
    c_im: f64x4,
    max_iter: u32,
) -> [u32; 4] {
    let zero = f64x4::splat(0.0);
    let one = f64x4::splat(1.0);
    let two = f64x4::splat(2.0);
    let four = f64x4::splat(4.0);
    let quarter = f64x4::splat(0.25);
    let point0625 = f64x4::splat(0.0625);

    let c_re_shifted = c_re - quarter;
    let q = c_re_shifted * c_re_shifted + c_im * c_im;
    let in_cardioid = q.simd_le(q * c_re_shifted + quarter * c_im * c_im);

    let c_re_plus1 = c_re + one;
    let in_bulb = (c_re_plus1 * c_re_plus1 + c_im * c_im).simd_le(point0625);

    let skip = in_cardioid | in_bulb;

    let mut z_re = zero;
    let mut z_im = zero;
    let mut iter_count = u32x4::splat(max_iter);
    let mut active: Mask<i64, 4> = !skip;

    if !active.any() {
        return iter_count.to_array();
    }

    for i in 0..max_iter {
        let z_re_sq = z_re * z_re;
        let z_im_sq = z_im * z_im;
        let mag_sq = z_re_sq + z_im_sq;

        let escaped = mag_sq.simd_gt(four);
        let newly_escaped = escaped & active;

        if newly_escaped.any() {
            iter_count = newly_escaped.select(u32x4::splat(i), iter_count);
        }

        active = active & !escaped;

        if !active.any() {
            break;
        }

        let z_im_new = two * z_re * z_im + c_im;
        let z_re_new = z_re_sq - z_im_sq + c_re;
        z_re = active.select(z_re_new, z_re);
        z_im = active.select(z_im_new, z_im);
    }

    iter_count.to_array()
}

/// SIMD 渲染器（带 LUT 缓存）
pub struct SimdRenderer {
    width: u32,
    height: u32,
    color_scheme: ColorScheme,
    aspect_ratio: f64,
    cached_lut: Vec<[u8; 3]>,
    cached_max_iter: u32,
}

impl SimdRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme) -> Self {
        Self {
            width,
            height,
            color_scheme,
            aspect_ratio: width as f64 / height as f64,
            cached_lut: Vec::new(),
            cached_max_iter: 0,
        }
    }
}

impl Renderer for SimdRenderer {
    fn render(&mut self, state: &ZoomState) -> anyhow::Result<Vec<u8>> {
        let anim = ZoomAnimation::new(
            &crate::config::ZoomConfig::default(),
            1,
            state.max_iter,
        );

        let (x_min, x_max, y_min, y_max) = anim.get_view_bounds(state, self.aspect_ratio);

        let width = self.width as usize;
        let height = self.height as usize;
        let max_iter = state.max_iter;

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let pixel_width = x_range / self.width as f64;
        let pixel_height = y_range / self.height as f64;

        // LUT 缓存：max_iter 不变时复用
        if max_iter != self.cached_max_iter || self.cached_lut.is_empty() {
            self.cached_lut = generate_color_lut(max_iter, self.color_scheme);
            self.cached_max_iter = max_iter;
        }
        let lut = &self.cached_lut;

        let mut raw_pixels = vec![0u8; width * height * 3];

        let simd_width = width / 4;
        let remainder = width % 4;

        raw_pixels
            .par_chunks_mut(width * 3)
            .enumerate()
            .for_each(|(y, row_slice)| {
                let c_im = y_min + y as f64 * pixel_height;

                for sx in 0..simd_width {
                    let x_base = sx * 4;
                    let c_re = f64x4::from_array([
                        x_min + (x_base) as f64 * pixel_width,
                        x_min + (x_base + 1) as f64 * pixel_width,
                        x_min + (x_base + 2) as f64 * pixel_width,
                        x_min + (x_base + 3) as f64 * pixel_width,
                    ]);
                    let c_im_vec = f64x4::splat(c_im);

                    let iters = mandelbrot_simd_f64x4(c_re, c_im_vec, max_iter);

                    for lane in 0..4 {
                        let x = x_base + lane;
                        let color = &lut[iters[lane] as usize];
                        let idx = x * 3;
                        row_slice[idx] = color[0];
                        row_slice[idx + 1] = color[1];
                        row_slice[idx + 2] = color[2];
                    }
                }

                for x in simd_width * 4..simd_width * 4 + remainder {
                    let c_re = x_min + x as f64 * pixel_width;
                    let iter = mandelbrot(c_re, c_im, max_iter) as usize;
                    let color = &lut[iter];
                    let idx = x * 3;
                    row_slice[idx] = color[0];
                    row_slice[idx + 1] = color[1];
                    row_slice[idx + 2] = color[2];
                }
            });

        Ok(raw_pixels)
    }

    fn name(&self) -> &'static str {
        "CPU (SIMD)"
    }
}

/// SIMD 批量渲染器
pub struct SimdBatchRenderer {
    renderer: SimdRenderer,
    progress_bar: ProgressBar,
}

impl SimdBatchRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme, total_frames: u32) -> Self {
        let renderer = SimdRenderer::new(width, height, color_scheme);
        let progress_bar = ProgressBar::new(total_frames as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} 帧 ({eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );
        Self { renderer, progress_bar }
    }

    pub fn render_frame(&mut self, state: &ZoomState) -> anyhow::Result<Vec<u8>> {
        let result = self.renderer.render(state);
        self.progress_bar.inc(1);
        result
    }

    pub fn finish(&self) {
        self.progress_bar.finish_with_message("渲染完成!");
    }
}
