# Mandelbrot 分形视频生成器

高性能 Mandelbrot 分形缩放视频生成器，支持 GPU/CPU 自动切换，输出高质量视频。

## 特性

- **自动 GPU/CPU 切换**：检测 GPU 的 f64 精度支持，自动选择最佳渲染模式
- **GPU 加速**：使用 wgpu compute shader 进行 GPU 计算（支持 f64 时）
- **CPU 并行**：使用 Rayon 多线程并行计算（GPU 不支持 f64 时回退）
- **优化算法**：主心形检测、周期2圆盘检测、周期性检测
- **多种编码格式**：支持 MP4 (H.264)、WebM (VP9)、IVF (AV1)、MKV
- **预设位置**：内置多个经典分形位置（海马谷、大象谷、三重螺旋等）
- **可配置**：通过 TOML 配置文件自定义所有参数

## 依赖

- Rust 1.70+
- FFmpeg（用于视频编码）

## 安装

```bash
git clone <repo>
cd mandelbro_video
cargo build --release
```

## 使用

```bash
# 使用默认配置
cargo run --release

# 使用指定配置文件
cargo run --release -- test_config.toml
```

## 配置文件

```toml
[video]
width = 1920           # 分辨率宽度
height = 1080          # 分辨率高度
fps = 30               # 帧率
duration_seconds = 60  # 视频时长（秒）
output = "mandelbrot_zoom.mp4"  # 输出文件

[zoom]
# 预设位置: Seahorse, Elephant, TripleSpiral, MiniMandelbrot, Custom
preset = "Seahorse"
# 自定义位置时使用以下参数:
# center_re = -0.743643887037151
# center_im = 0.131825904205330
# initial_zoom = 1.0
# final_zoom = 1e14

[render]
max_iter_base = 256    # 基础迭代次数
color_scheme = "Classic"  # 颜色方案: Classic, Fire, Ocean, Rainbow, Electric, Grayscale
force_cpu = false      # 强制使用 CPU

[encoding]
quantizer = 50         # 质量参数（越低质量越高，0-255）
```

## 预设位置

| 预设 | 位置 | 最终缩放 |
|------|------|----------|
| Seahorse | (-0.7436, 0.1318) | 10^14 |
| Elephant | (0.275, 0.0) | 10^10 |
| TripleSpiral | (-0.088, 0.654) | 10^12 |
| MiniMandelbrot | (-1.768, 0.001) | 10^15 |

## 输出格式

| 扩展名 | 编码器 | 说明 |
|--------|--------|------|
| .mp4 | H.264 (libx264) | 通用格式，兼容性最好 |
| .webm | VP9 (libvpx-vp9) | 网页格式 |
| .mkv | H.264 | 高质量存档 |
| .ivf | AV1 (libaom-av1) | AV1 测试格式 |

## 性能

- 1080p 60秒视频（1800帧）
- CPU 模式（Rayon 多线程）：约 5-10 分钟
- GPU 模式（f64 shader）：约 1-3 分钟

## 技术架构

```
src/
├── main.rs           # 入口，GPU 检测，主循环
├── config.rs         # 配置结构和预设
├── renderer/
│   ├── mod.rs        # Renderer trait，颜色映射
│   ├── gpu.rs        # wgpu GPU 实现
│   └── cpu.rs        # Rayon CPU 实现
├── encoder.rs        # FFmpeg 管道编码
└── zoom.rs           # 缩放动画计算
```

## 许可证

MIT
