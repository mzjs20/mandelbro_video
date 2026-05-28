#![feature(portable_simd)]
mod config;
mod renderer;
mod encoder;
mod zoom;

use config::Config;
use renderer::{Renderer, gpu::GpuRenderer, cpu::SimdBatchRenderer};
use encoder::Av1Encoder;
use zoom::ZoomAnimation;
use indicatif::{ProgressBar, ProgressStyle};
use anyhow::{Result, Context};

fn main() -> Result<()> {
    // 初始化日志
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // 加载配置
    let args: Vec<String> = std::env::args().collect();
    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "config.toml"
    };

    let config = Config::from_file(config_path)
        .with_context(|| format!("加载配置文件 {} 失败", config_path))?;

    log::info!("配置加载成功:");
    log::info!("  分辨率: {}x{}", config.video.width, config.video.height);
    log::info!("  FPS: {}", config.video.fps);
    log::info!("  时长: {}秒", config.video.duration_seconds);
    log::info!("  输出: {}", config.video.output);

    let (center_re, center_im, initial_zoom, final_zoom) = config.zoom.get_actual_params();
    log::info!("  缩放目标: ({}, {})", center_re, center_im);
    log::info!("  缩放范围: {} -> {}", initial_zoom, final_zoom);

    // 计算总帧数
    let total_frames = config.video.fps * config.video.duration_seconds;

    // 创建缩放动画
    let zoom_anim = ZoomAnimation::new(&config.zoom, total_frames, config.render.max_iter_base);

    // 检测GPU f64支持
    let use_gpu = if config.render.force_cpu {
        log::info!("强制使用CPU模式");
        false
    } else {
        match GpuRenderer::check_f64_support() {
            Ok(supports) => {
                if supports {
                    log::info!("GPU支持f64精度，使用GPU模式");
                    true
                } else {
                    log::info!("GPU不支持f64精度，使用CPU模式以保证精度");
                    false
                }
            }
            Err(e) => {
                log::warn!("GPU检测失败: {}, 使用CPU模式", e);
                false
            }
        }
    };

    // 创建渲染器
    if use_gpu {
        render_with_gpu(&config, zoom_anim, total_frames)?;
    } else {
        render_with_cpu(&config, zoom_anim, total_frames)?;
    }

    log::info!("视频生成完成!");
    Ok(())
}

fn render_with_gpu(config: &Config, zoom_anim: ZoomAnimation, total_frames: u32) -> Result<()> {
    let mut renderer = GpuRenderer::new(
        config.video.width,
        config.video.height,
        config.render.color_scheme,
    ).context("创建GPU渲染器失败")?;

    log::info!("渲染器: {}", renderer.name());

    // 进度条
    let progress = ProgressBar::new(total_frames as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} 帧 ({eta}) GPU渲染")
            .unwrap()
            .progress_chars("█▓░"),
    );

    // 收集所有帧
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(total_frames as usize);

    for frame_idx in 0..total_frames {
        let state = zoom_anim.get_frame(frame_idx);
        let frame_data = renderer.render(&state)
            .with_context(|| format!("渲染帧 {} 失败", frame_idx))?;
        frames.push(frame_data);
        progress.inc(1);
    }

    progress.finish_with_message("渲染完成，开始编码...");

    // 编码视频
    let mut encoder = Av1Encoder::new(
        config.video.width,
        config.video.height,
        config.video.fps,
        total_frames,
        config.video.output.clone(),
        config.encoding.clone(),
    ).context("创建编码器失败")?;

    encoder.encode(frames)?;

    Ok(())
}

fn render_with_cpu(config: &Config, zoom_anim: ZoomAnimation, total_frames: u32) -> Result<()> {
    let mut batch_renderer = SimdBatchRenderer::new(
        config.video.width,
        config.video.height,
        config.render.color_scheme,
        total_frames,
    );

    log::info!("渲染器: CPU (SIMD + Rayon多线程)");

    // 收集所有帧
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(total_frames as usize);

    for frame_idx in 0..total_frames {
        let state = zoom_anim.get_frame(frame_idx);
        let frame_data = batch_renderer.render_frame(&state)
            .with_context(|| format!("渲染帧 {} 失败", frame_idx))?;
        frames.push(frame_data);
    }

    batch_renderer.finish();

    // 编码视频
    log::info!("开始编码AV1视频...");
    let mut encoder = Av1Encoder::new(
        config.video.width,
        config.video.height,
        config.video.fps,
        total_frames,
        config.video.output.clone(),
        config.encoding.clone(),
    ).context("创建编码器失败")?;

    encoder.encode(frames)?;

    Ok(())
}