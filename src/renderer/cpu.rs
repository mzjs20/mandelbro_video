use crate::config::ColorScheme;
use crate::zoom::{ZoomState, ZoomAnimation};
use crate::renderer::{Renderer, generate_color_lut};
use crate::renderer::perturbation::{
    ReferenceOrbit, suggested_precision,
    SeriesApproximation, perturbation_iterate_with_series,
    perturbation_iterate_with_series_simd,
    perturbation_iterate_simd, perturbation_iterate,
};
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

/// 低缩放级别的朴素 SIMD 渲染（用于扰动法的 fallback）
fn render_naive_simd(
    width: u32, height: u32, max_iter: u32,
    x_min: f64, y_min: f64, pixel_w: f64, pixel_h: f64,
    lut: &[[u8; 3]],
) -> anyhow::Result<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let mut raw_pixels = vec![0u8; w * h * 3];

    raw_pixels
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row_slice)| {
            let c_im = y_min + y as f64 * pixel_h;
            let simd_width = w / 4;

            for sx in 0..simd_width {
                let x_base = sx * 4;
                let c_re = f64x4::from_array([
                    x_min + x_base as f64 * pixel_w,
                    x_min + (x_base + 1) as f64 * pixel_w,
                    x_min + (x_base + 2) as f64 * pixel_w,
                    x_min + (x_base + 3) as f64 * pixel_w,
                ]);
                let iters = mandelbrot_simd_f64x4(c_re, f64x4::splat(c_im), max_iter);
                for lane in 0..4 {
                    let color = &lut[iters[lane] as usize];
                    let idx = (x_base + lane) * 3;
                    row_slice[idx] = color[0];
                    row_slice[idx + 1] = color[1];
                    row_slice[idx + 2] = color[2];
                }
            }

            for x in simd_width * 4..w {
                let c_re = x_min + x as f64 * pixel_w;
                let iter = mandelbrot(c_re, c_im, max_iter);
                let color = &lut[iter as usize];
                let idx = x * 3;
                row_slice[idx] = color[0];
                row_slice[idx + 1] = color[1];
                row_slice[idx + 2] = color[2];
            }
        });

    Ok(raw_pixels)
}

/// 扰动理论渲染器：用参考轨道 + ε 迭代加速深度缩放渲染
pub struct PerturbationRenderer {
    width: u32,
    height: u32,
    color_scheme: ColorScheme,
    aspect_ratio: f64,
    cached_lut: Vec<[u8; 3]>,
    cached_max_iter: u32,
    /// 缓存的参考轨道（帧间复用）
    cached_orbit: Option<ReferenceOrbit>,
    /// 后台预计算的参考轨道
    bg_orbit: Option<std::thread::JoinHandle<ReferenceOrbit>>,
    /// 缓存的级数近似（系数不变，每帧只重算 skip）
    cached_series: Option<SeriesApproximation>,
    /// 缩放目标中心坐标（预计算参考轨道用）
    center_re: f64,
    center_im: f64,
    /// 最大 max_iter（预计算参考轨道用）
    max_orbit_iter: u32,
    /// 最终缩放级别（用于确定精度）
    final_zoom: f64,
    /// 最近的级数近似信息 (skip, max_iter)，供进度条显示
    pub last_series_info: (u32, u32),
}

impl PerturbationRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme, max_orbit_iter: u32, final_zoom: f64, center_re: f64, center_im: f64) -> Self {
        // 在后台线程预计算参考轨道，避免渲染时阻塞
        let precision = suggested_precision(final_zoom);
        log::info!("后台预计算参考轨道: c=({:.15}, {:.15}), max_iter={}, prec={}", center_re, center_im, max_orbit_iter, precision);
        let bg_orbit = std::thread::spawn(move || {
            ReferenceOrbit::compute(center_re, center_im, max_orbit_iter, precision)
        });

        Self {
            width,
            height,
            color_scheme,
            aspect_ratio: width as f64 / height as f64,
            cached_lut: Vec::new(),
            cached_max_iter: 0,
            cached_orbit: None,
            bg_orbit: Some(bg_orbit),
            cached_series: None,
            center_re,
            center_im,
            max_orbit_iter,
            final_zoom,
            last_series_info: (0, 0),
        }
    }
}

impl Renderer for PerturbationRenderer {
    fn render(&mut self, state: &ZoomState) -> anyhow::Result<Vec<u8>> {
        let anim = ZoomAnimation::new(
            &crate::config::ZoomConfig::default(),
            1,
            state.max_iter,
        );

        let (x_min, x_max, y_min, y_max) = anim.get_view_bounds(state, self.aspect_ratio);

        let width = self.width;
        let height = self.height;
        let max_iter = state.max_iter;

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let pixel_width = x_range / width as f64;
        let pixel_height = y_range / height as f64;

        // LUT 缓存
        if max_iter != self.cached_max_iter || self.cached_lut.is_empty() {
            self.cached_lut = generate_color_lut(max_iter, self.color_scheme);
            self.cached_max_iter = max_iter;
        }
        let lut = &self.cached_lut;

        // 混合策略：低缩放时用朴素 SIMD，高缩放时用扰动理论
        // f64 在 zoom < 1e12 时精度足够（δ/eps > 1000），朴素 SIMD 又快又准
        // zoom ≥ 1e12 时 f64 像素坐标开始量化，必须用扰动法保持精度
        // 且此时级数近似 skip 比例高，能补偿 ε 迭代的额外算术开销
        if state.zoom < 1e12 {
            return render_naive_simd(width, height, max_iter, x_min, y_min, pixel_width, pixel_height, lut);
        }

        // 高缩放：扰动理论渲染
        if self.cached_orbit.is_none() {
            // 从后台线程获取参考轨道
            if let Some(handle) = self.bg_orbit.take() {
                log::info!("等待参考轨道计算完成...");
                self.cached_orbit = Some(handle.join().expect("参考轨道计算线程崩溃"));
                log::info!("参考轨道就绪");
            }
        }
        let orbit = self.cached_orbit.as_ref().unwrap();

        // 计算或更新级数近似
        let view_height = 4.0 / state.zoom;
        let delta_max = view_height * 1.5;

        if self.cached_series.is_none() {
            self.cached_series = Some(SeriesApproximation::compute(orbit, delta_max, 32, max_iter));
        } else {
            self.cached_series.as_mut().unwrap().recompute_skip(delta_max, max_iter);
        }
        let series = self.cached_series.as_ref().unwrap();
        let skip = series.skip_iterations();
        self.last_series_info = (skip, max_iter);

        // 并行 SIMD 渲染，直接生成 RGB
        let w = width as usize;
        let simd_width = w / 4;
        let remainder = w % 4;

        let mut raw_pixels = vec![0u8; (width * height * 3) as usize];

        raw_pixels
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row_slice)| {
                let c_im = y_min + y as f64 * pixel_height;
                let c_im_vec = f64x4::splat(c_im);

                // SIMD: 每次处理 4 个像素
                for sx in 0..simd_width {
                    let x_base = sx * 4;
                    let c_re_vec = f64x4::from_array([
                        x_min + x_base as f64 * pixel_width,
                        x_min + (x_base + 1) as f64 * pixel_width,
                        x_min + (x_base + 2) as f64 * pixel_width,
                        x_min + (x_base + 3) as f64 * pixel_width,
                    ]);

                    let iters = if skip > 0 {
                        perturbation_iterate_with_series_simd(
                            orbit, &series, c_re_vec, c_im_vec, max_iter,
                        )
                    } else {
                        perturbation_iterate_simd(
                            orbit, c_re_vec, c_im_vec, max_iter,
                        )
                    };

                    for lane in 0..4 {
                        let color = &lut[iters[lane] as usize];
                        let idx = (x_base + lane) * 3;
                        row_slice[idx] = color[0];
                        row_slice[idx + 1] = color[1];
                        row_slice[idx + 2] = color[2];
                    }
                }

                // 余数像素用标量
                for x in simd_width * 4..simd_width * 4 + remainder {
                    let c_re = x_min + x as f64 * pixel_width;
                    let iter = if skip > 0 {
                        perturbation_iterate_with_series(
                            orbit, &series, c_re, c_im, max_iter,
                        )
                    } else {
                        perturbation_iterate(
                            orbit, c_re, c_im, max_iter,
                        )
                    };
                    let color = &lut[iter as usize];
                    let idx = x * 3;
                    row_slice[idx] = color[0];
                    row_slice[idx + 1] = color[1];
                    row_slice[idx + 2] = color[2];
                }
            });

        Ok(raw_pixels)
    }

    fn name(&self) -> &'static str {
        "CPU (Perturbation)"
    }
}

/// 扰动理论批量渲染器
pub struct PerturbationBatchRenderer {
    renderer: PerturbationRenderer,
    progress_bar: ProgressBar,
    last_skip: u32,
    last_max_iter: u32,
}

impl PerturbationBatchRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme, total_frames: u32, max_orbit_iter: u32, final_zoom: f64, center_re: f64, center_im: f64) -> Self {
        let renderer = PerturbationRenderer::new(width, height, color_scheme, max_orbit_iter, final_zoom, center_re, center_im);
        let progress_bar = ProgressBar::new(total_frames as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} 帧 ({eta}) 扰动渲染|{msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        Self { renderer, progress_bar, last_skip: 0, last_max_iter: 0 }
    }

    pub fn render_frame(&mut self, state: &ZoomState) -> anyhow::Result<Vec<u8>> {
        let result = self.renderer.render(state);
        if self.renderer.last_series_info != (self.last_skip, self.last_max_iter) {
            let (skip, max_iter) = self.renderer.last_series_info;
            if skip > 0 {
                self.progress_bar.set_message(format!("跳过 {}/{} 迭代", skip, max_iter));
            } else {
                self.progress_bar.set_message("朴素 SIMD");
            }
            self.last_skip = skip;
            self.last_max_iter = max_iter;
        }
        self.progress_bar.inc(1);
        result
    }

    pub fn finish(&self) {
        self.progress_bar.finish_with_message("扰动渲染完成!");
    }
}
