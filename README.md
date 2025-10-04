<<<<<<< HEAD
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
=======
# 曼德博集合视频生成工具

一个高性能的曼德博集合（Mandelbrot Set）分形视频生成工具，利用GPU加速计算和并行处理技术，能够生成高质量的分形动画。

## 功能特点

- **GPU加速计算**：使用Metal框架进行GPU加速，大幅提升分形计算速度
- **多种颜色方案**：内置经典、火焰、平滑等多种颜色映射方案，支持自定义扩展
- **智能并行处理**：根据图像大小自动选择最佳处理模式（并行/串行）
- **高度可配置**：通过配置文件调整分形参数、视频属性和颜色效果
- **平滑动画过渡**：采用平滑迭代算法，避免帧间颜色闪烁，生成流畅动画

## 安装要求

- 操作系统：macOS（需要Metal支持）
- 依赖工具：
    - Rust 1.60+ 开发环境
    - FFmpeg（用于视频编码）
    - Xcode命令行工具（提供Metal编译支持）

## 安装步骤

1. **克隆仓库**
>>>>>>> f79c46c80466dce9e63b924f418177179b51ee59
   ```bash
   git clone https://github.com/mzjs20/mandelbro_video.git
   cd mandelbro_video
   ```

<<<<<<< HEAD
2. 构建项目：
=======
2. **安装依赖**
   ```bash
   # 安装FFmpeg（如果未安装）
   brew install ffmpeg
   
   # 安装Xcode命令行工具（可选）
   xcode-select --install
   ```

3. **构建项目**
>>>>>>> f79c46c80466dce9e63b924f418177179b51ee59
   ```bash
   cargo build --release
   ```

<<<<<<< HEAD
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
=======
   编译后的可执行文件位于 `target/release/mandelbrot_video`

## 使用方法

1. **配置参数**

   复制并修改示例配置文件：
   ```bash
   cp config.example.toml config.toml
   ```

   编辑 `config.toml` 文件调整参数（详见配置说明）

2. **生成视频**
   ```bash
   ./target/release/mandelbrot_video
   ```

3. **查看结果**

   生成的视频将保存到配置文件指定的 `output_file` 路径，临时帧文件会自动清理

## 配置说明

`config.toml` 文件包含以下配置项：

### 视频基本设置
```toml
width = 1920          # 视频宽度（像素）
height = 1080         # 视频高度（像素）
duration = 10.0       # 视频时长（秒）
fps = 30              # 帧率
output_file = "mandelbrot.mp4"  # 输出视频路径
temp_frame_dir = "tmp_frames"   # 临时帧存储目录
```

### 分形参数设置
```toml
center_x = -0.5       # 初始X轴中心坐标
center_y = 0.0        # 初始Y轴中心坐标
start_zoom = 1.0      # 初始缩放比例
zoom_rate = 1.5       # 缩放速率（>1.0为放大，<1.0为缩小）
max_iterations = 1000 # 最大迭代次数
iteration_growth = 1.2 # 迭代次数增长因子
```

### 颜色设置
```toml
color_palette = "fire" # 颜色方案：classic, fire, smooth
color_smoothing = 0.8  # 颜色平滑因子（0.0-1.0）
```

### 并行处理设置
```toml
parallel_color_mapping = true  # 是否启用并行颜色映射
parallel_threshold_pixels = 1000000  # 启用并行的像素阈值
```

## 颜色方案说明

- `classic`：经典分形配色，从蓝色到红色的连续色调变化
- `fire`：火焰风格，从黑色到红色、黄色的渐变
- `smooth`：平滑过渡的蓝紫色调，适合展现细节

可以通过修改代码轻松添加自定义颜色方案（参见源码中的`palette`函数）

## 性能优化建议

1. **调整并行阈值**：根据CPU核心数调整`parallel_threshold_pixels`，通常4K分辨率建议启用并行
2. **平衡质量与速度**：降低`max_iterations`可提高速度，但会损失细节
3. **合理设置帧率**：一般场景30fps已足够，动态变化剧烈时可提高到60fps
4. **预编译优化**：使用`cargo build --release`获得最佳性能

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件
>>>>>>> f79c46c80466dce9e63b924f418177179b51ee59
