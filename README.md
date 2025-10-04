# 曼德勃罗集视频生成器

一个使用 Rust 和 wgpu 加速的曼德勃罗集分形视频生成工具，能够生成高质量的缩放动画效果。

## 功能特点

- 利用 GPU 加速计算，快速生成高分辨率曼德勃罗集图像
- 支持自定义缩放路径、迭代次数和色彩方案
- 自动生成平滑的缩放动画并导出为 MP4 视频
- 可配置的分辨率、帧率和总帧数

## 安装要求

- Rust 1.60+ 开发环境
- FFmpeg（用于视频编码，需要添加到系统 PATH 中）
- 支持 WebGPU 的显卡（主流 NVIDIA、AMD、Intel 显卡均可）

## 安装步骤

1. 克隆仓库（或下载源代码）：
   ```bash
   git clone https://github.com/mzjs20/mandelbro_video.git
   cd mandelbro_video
   ```

2. 构建项目：
   ```bash
   cargo build --release
   ```

## 使用方法

1. 配置参数（修改 `config.toml` 文件）：
   ```toml
   [rendering]
   width = 3840       # 视频宽度
   height = 2160      # 视频高度
   fps = 60           # 帧率
   total_frames = 60  # 总帧数

   [mandelbrot]
   center_x = -0.743643887037151  # 初始中心X坐标
   center_y = 0.131825904205330   # 初始中心Y坐标
   initial_radius = 0.005         # 初始缩放半径
   zoom_factor = 1.01             # 每帧缩放因子
   initial_iterations = 2000      # 初始迭代次数
   max_iterations = 100000        # 最大迭代次数
   iteration_growth_factor = 1.01 # 迭代次数增长因子
   ```

2. 运行程序：
   ```bash
   cargo run --release
   ```

3. 生成的视频将保存为 `mandelbrot_zoom.mp4`

## 配置说明

- **渲染设置（rendering）**：
    - `width` 和 `height`：输出视频的分辨率
    - `fps`：视频帧率（每秒帧数）
    - `total_frames`：视频总帧数

- **曼德勃罗集设置（mandelbrot）**：
    - `center_x`, `center_y`：初始视图中心坐标
    - `initial_radius`：初始缩放半径（值越小，放大倍数越高）
    - `zoom_factor`：每帧缩放因子（>1 表示放大，<1 表示缩小）
    - `initial_iterations`：起始迭代次数
    - `max_iterations`：最大迭代次数（防止计算量过大）
    - `iteration_growth_factor`：迭代次数增长因子（随缩放自动增加迭代次数）

## 性能提示

- 更高的分辨率和更多的帧数会显著增加计算时间
- 迭代次数越多，图像细节越丰富，但计算时间也越长
- 对于非常高的分辨率（如 4K），建议减少总帧数或降低迭代次数
- 程序会自动利用 GPU 加速，确保你的显卡驱动是最新版本