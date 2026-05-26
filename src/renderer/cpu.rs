use crate::config::ColorScheme;
use crate::zoom::{ZoomState, ZoomAnimation};
use crate::renderer::{Renderer, generate_color_lut};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

/// CPU渲染器
pub struct CpuRenderer {
    width: u32,
    height: u32,
    color_scheme: ColorScheme,
    aspect_ratio: f64,
}

impl CpuRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme) -> Self {
        Self {
            width,
            height,
            color_scheme,
            aspect_ratio: width as f64 / height as f64,
        }
    }
}

impl Renderer for CpuRenderer {
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

        // 预计算颜色查找表
        let lut = generate_color_lut(max_iter, self.color_scheme);

        // 预分配像素内存
        let mut raw_pixels = vec![0u8; width * height * 3];

        // 粗粒度并行处理
        let chunk_rows: usize = 8;
        let chunk_size = width * 3 * chunk_rows;

        raw_pixels
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk_slice)| {
                let start_y = chunk_idx * chunk_rows;
                let actual_rows = chunk_slice.len() / (width * 3);

                for y_offset in 0..actual_rows {
                    let y = start_y + y_offset;
                    let c_im = y_min + y as f64 * pixel_height;
                    let row_start = y_offset * (width * 3);

                    for x in 0..width {
                        let c_re = x_min + x as f64 * pixel_width;
                        let iter = mandelbrot(c_re, c_im, max_iter) as usize;
                        let color = &lut[iter];

                        let idx = row_start + x * 3;
                        chunk_slice[idx] = color[0];
                        chunk_slice[idx + 1] = color[1];
                        chunk_slice[idx + 2] = color[2];
                    }
                }
            });

        Ok(raw_pixels)
    }

    fn name(&self) -> &'static str {
        "CPU (Rayon)"
    }
}

/// 优化的Mandelbrot计算（主心形检测 + 周期性检测）
#[inline(always)]
fn mandelbrot(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    // 主心形检测
    let q = (c_re - 0.25).powi(2) + c_im.powi(2);
    if q * (q + (c_re - 0.25)) <= 0.25 * c_im.powi(2) {
        return max_iter;
    }

    // 周期2圆盘检测
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

        // 周期性检测
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

/// 批量渲染多帧（用于视频生成）
pub struct CpuBatchRenderer {
    renderer: CpuRenderer,
    progress_bar: ProgressBar,
}

impl CpuBatchRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme, total_frames: u32) -> Self {
        let renderer = CpuRenderer::new(width, height, color_scheme);
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