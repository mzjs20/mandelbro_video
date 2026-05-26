/// 缩放动画计算
use crate::config::ZoomConfig;

/// 表示一帧的缩放状态
#[derive(Debug, Clone, Copy)]
pub struct ZoomState {
    /// 中心点实部
    pub center_re: f64,
    /// 中心点虚部
    pub center_im: f64,
    /// 缩放级别（值越大，视野越小）
    pub zoom: f64,
    /// 当前帧的建议最大迭代次数
    pub max_iter: u32,
}

/// 缩放动画生成器
pub struct ZoomAnimation {
    center_re: f64,
    center_im: f64,
    initial_zoom: f64,
    final_zoom: f64,
    total_frames: u32,
    base_max_iter: u32,
}

impl ZoomAnimation {
    pub fn new(zoom_config: &ZoomConfig, total_frames: u32, base_max_iter: u32) -> Self {
        let (center_re, center_im, initial_zoom, final_zoom) = zoom_config.get_actual_params();

        Self {
            center_re,
            center_im,
            initial_zoom,
            final_zoom,
            total_frames,
            base_max_iter,
        }
    }

    /// 计算指定帧的缩放状态
    pub fn get_frame(&self, frame: u32) -> ZoomState {
        let t = frame as f64 / (self.total_frames - 1).max(1) as f64;

        // 指数缩放：平滑过渡
        let zoom = self.initial_zoom * (self.final_zoom / self.initial_zoom).powf(t);

        // 根据缩放级别动态调整迭代次数
        // 缩放越大，需要越多迭代来显示细节
        let zoom_factor = zoom.log10().max(0.0);
        let max_iter = (self.base_max_iter as f64 * (1.0 + zoom_factor * 0.5)) as u32;
        let max_iter = max_iter.min(10000); // 设置上限避免过慢

        ZoomState {
            center_re: self.center_re,
            center_im: self.center_im,
            zoom,
            max_iter,
        }
    }

    /// 计算指定缩放级别下的视图范围
    pub fn get_view_bounds(&self, state: &ZoomState, aspect_ratio: f64) -> (f64, f64, f64, f64) {
        // 缩放级别表示每单位像素对应的复平面距离
        // zoom=1 时，视图范围为 [-2, 2] 左右
        let view_size = 4.0 / state.zoom;

        let half_width = view_size / 2.0;
        let half_height = half_width / aspect_ratio;

        let x_min = state.center_re - half_width;
        let x_max = state.center_re + half_width;
        let y_min = state.center_im - half_height;
        let y_max = state.center_im + half_height;

        (x_min, x_max, y_min, y_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zoom_animation() {
        let zoom_config = ZoomConfig::default();
        let anim = ZoomAnimation::new(&zoom_config, 100, 256);

        let first = anim.get_frame(0);
        let last = anim.get_frame(99);

        assert!(first.zoom < last.zoom);
        assert!(first.max_iter <= last.max_iter);
    }
}
