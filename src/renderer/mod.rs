pub mod cpu;
pub mod gpu;
pub mod perturbation;

use anyhow::Result;
use crate::config::ColorScheme;
use crate::zoom::ZoomState;

/// 渲染器trait
pub trait Renderer {
    /// 渲染一帧
    fn render(&mut self, state: &ZoomState) -> Result<Vec<u8>>;

    /// 获取渲染器名称
    fn name(&self) -> &'static str;
}

/// 颜色映射函数
#[inline(always)]
pub fn iter_to_color(iter: u32, max_iter: u32, scheme: ColorScheme) -> [u8; 3] {
    if iter == max_iter {
        return [0, 0, 0];
    }

    let t = iter as f64 / max_iter as f64;

    match scheme {
        ColorScheme::Classic => {
            // 经典蓝紫色调
            let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u8;
            let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8;
            let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8;
            [r, g, b]
        }
        ColorScheme::Fire => {
            // 火焰色调
            let r = if t < 0.33 { (t * 3.0 * 255.0) as u8 } else { 255 };
            let g = if t < 0.33 { 0 } else if t < 0.66 { ((t - 0.33) * 3.0 * 255.0) as u8 } else { 255 };
            let b = if t < 0.66 { 0 } else { ((t - 0.66) * 3.0 * 255.0) as u8 };
            [r, g, b]
        }
        ColorScheme::Ocean => {
            // 海洋色调
            let r = (t * 0.3 * 255.0) as u8;
            let g = (t * 0.7 * 255.0) as u8;
            let b = (0.3 + t * 0.7 * 255.0) as u8;
            [r, g, b]
        }
        ColorScheme::Rainbow => {
            // 彩虹色调
            let hue = t * 360.0;
            hsv_to_rgb(hue, 1.0, 1.0)
        }
        ColorScheme::Electric => {
            // 电子色调（高对比）
            let v = (t * 255.0) as u8;
            let r = v;
            let g = ((v as f64 * 0.5).min(255.0)) as u8;
            let b = ((255.0 - v as f64 * 0.3).max(0.0)) as u8;
            [r, g, b]
        }
        ColorScheme::Grayscale => {
            let v = (t * 255.0) as u8;
            [v, v, v]
        }
    }
}

/// HSV转RGB
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

/// 生成颜色查找表
pub fn generate_color_lut(max_iter: u32, scheme: ColorScheme) -> Vec<[u8; 3]> {
    (0..=max_iter)
        .map(|i| iter_to_color(i, max_iter, scheme))
        .collect()
}