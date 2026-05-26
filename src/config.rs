use serde::Deserialize;

/// 预设缩放位置
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum ZoomPreset {
    /// 海马谷 - 经典美丽位置
    Seahorse,
    /// 大象谷 - 另一个经典位置
    Elephant,
    /// 三重螺旋
    TripleSpiral,
    /// 迷你Mandelbrot
    MiniMandelbrot,
    /// 自定义位置
    Custom,
}

impl Default for ZoomPreset {
    fn default() -> Self {
        Self::Seahorse
    }
}

impl ZoomPreset {
    /// 获取预设的中心坐标和推荐缩放范围
    pub fn get_params(&self) -> (f64, f64, f64, f64) {
        match self {
            // (center_re, center_im, initial_zoom, final_zoom)
            ZoomPreset::Seahorse => {
                (-0.743643887037151, 0.131825904205330, 1.0, 1e14)
            }
            ZoomPreset::Elephant => {
                (0.275, 0.0, 1.0, 1e10)
            }
            ZoomPreset::TripleSpiral => {
                (-0.088, 0.654, 1.0, 1e12)
            }
            ZoomPreset::MiniMandelbrot => {
                (-1.768, 0.001, 1.0, 1e15)
            }
            ZoomPreset::Custom => {
                // 默认值，会被配置文件覆盖
                (-0.5, 0.0, 1.0, 1e10)
            }
        }
    }
}

/// 颜色方案
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub enum ColorScheme {
    #[default]
    Classic,
    Fire,
    Ocean,
    Rainbow,
    Electric,
    Grayscale,
}

/// 视频配置
#[derive(Debug, Deserialize)]
pub struct VideoConfig {
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default = "default_duration")]
    pub duration_seconds: u32,
    #[serde(default = "default_output")]
    pub output: String,
}

fn default_width() -> u32 { 1920 }
fn default_height() -> u32 { 1080 }
fn default_fps() -> u32 { 30 }
fn default_duration() -> u32 { 60 }
fn default_output() -> String { "mandelbrot_zoom.ivf".to_string() }

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            width: default_width(),
            height: default_height(),
            fps: default_fps(),
            duration_seconds: default_duration(),
            output: default_output(),
        }
    }
}

/// 缩放配置
#[derive(Debug, Deserialize)]
pub struct ZoomConfig {
    #[serde(default)]
    pub preset: ZoomPreset,
    /// 自定义中心实部（仅preset=Custom时使用）
    pub center_re: Option<f64>,
    /// 自定义中心虚部（仅preset=Custom时使用）
    pub center_im: Option<f64>,
    /// 初始缩放级别
    pub initial_zoom: Option<f64>,
    /// 最终缩放级别
    pub final_zoom: Option<f64>,
}

impl Default for ZoomConfig {
    fn default() -> Self {
        Self {
            preset: ZoomPreset::default(),
            center_re: None,
            center_im: None,
            initial_zoom: None,
            final_zoom: None,
        }
    }
}

impl ZoomConfig {
    /// 获取实际的缩放参数
    pub fn get_actual_params(&self) -> (f64, f64, f64, f64) {
        let (preset_re, preset_im, preset_init, preset_final) = self.preset.get_params();

        let center_re = self.center_re.unwrap_or(preset_re);
        let center_im = self.center_im.unwrap_or(preset_im);
        let initial_zoom = self.initial_zoom.unwrap_or(preset_init);
        let final_zoom = self.final_zoom.unwrap_or(preset_final);

        (center_re, center_im, initial_zoom, final_zoom)
    }
}

/// 渲染配置
#[derive(Debug, Deserialize)]
pub struct RenderConfig {
    #[serde(default = "default_max_iter_base")]
    pub max_iter_base: u32,
    #[serde(default)]
    pub color_scheme: ColorScheme,
    /// 是否强制使用CPU（用于调试）
    #[serde(default)]
    pub force_cpu: bool,
}

fn default_max_iter_base() -> u32 { 256 }

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            max_iter_base: default_max_iter_base(),
            color_scheme: ColorScheme::default(),
            force_cpu: false,
        }
    }
}

/// 编码配置
#[derive(Debug, Clone, Deserialize)]
pub struct EncodingConfig {
    #[serde(default = "default_quantizer")]
    pub quantizer: u32,    // 质量参数，越低质量越高 (0-255)
}

fn default_quantizer() -> u32 { 50 }

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            quantizer: default_quantizer(),
        }
    }
}

/// 主配置结构
#[derive(Debug, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub video: VideoConfig,
    #[serde(default)]
    pub zoom: ZoomConfig,
    #[serde(default)]
    pub render: RenderConfig,
    #[serde(default)]
    pub encoding: EncodingConfig,
}

impl Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
