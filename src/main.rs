use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle, ProgressState};
use metal::*;
use objc::rc::autoreleasepool;
use rayon::prelude::*;
use serde::Deserialize;
use std::fmt;
use std::fs;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use std::os::raw::c_void;

// 配置结构定义 - 新增并行处理开关和阈值
#[derive(Debug, Deserialize)]
struct Config {
    width: usize,
    height: usize,
    duration: f64,
    fps: u32,
    output_file: String,
    center_x: f64,
    center_y: f64,
    start_zoom: f64,
    zoom_rate: f64,
    max_iterations: usize,
    iteration_growth: f64,
    color_palette: String,
    temp_frame_dir: String,
    color_smoothing: f64,
    // 新增：并行处理配置
    parallel_color_mapping: bool,       // 是否启用并行处理
    parallel_threshold_pixels: usize,   // 启用并行的像素数量阈值
}

// 验证配置参数有效性
fn validate_config(config: &Config) -> Result<()> {
    if config.width == 0 || config.height == 0 {
        anyhow::bail!("宽度和高度必须大于0");
    }
    if config.duration <= 0.0 {
        anyhow::bail!("视频时长必须大于0");
    }
    if config.fps == 0 {
        anyhow::bail!("帧率必须大于0");
    }
    if config.start_zoom <= 0.0 {
        anyhow::bail!("初始缩放比例必须大于0");
    }
    if config.zoom_rate <= 0.0 {
        anyhow::bail!("缩放率必须大于0");
    }
    if config.max_iterations == 0 {
        anyhow::bail!("最大迭代次数必须大于0");
    }
    if config.iteration_growth <= 0.0 {
        anyhow::bail!("迭代增长因子必须大于0");
    }
    if config.color_smoothing <= 0.0 || config.color_smoothing > 1.0 {
        anyhow::bail!("颜色平滑参数必须在0到1之间");
    }
    if config.parallel_threshold_pixels == 0 {
        anyhow::bail!("并行处理阈值像素数必须大于0");
    }
    Ok(())
}

// 从文件读取配置
fn load_config(path: &str) -> Result<Config> {
    let config_str = fs::read_to_string(path)
        .with_context(|| format!("无法读取配置文件: {}", path))?;

    let config = toml::from_str(&config_str)
        .with_context(|| "配置文件格式错误")?;

    validate_config(&config).context("配置参数无效")?;

    Ok(config)
}

// 初始化Metal设备和计算管道
fn init_metal() -> Result<(Device, ComputePipelineState)> {
    autoreleasepool(|| {
        let device = Device::system_default()
            .context("无法获取Metal设备，请确保您的Mac支持Metal")?;
        println!("✅ 成功获取Metal设备");

        let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void mandelbrot(
            const device float2* centers [[buffer(0)]],
            const device float* zoom [[buffer(1)]],
            const device uint* max_iter [[buffer(2)]],
            const device uint2* dimensions [[buffer(3)]],
            device float* iterations [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint width = dimensions->x;
            uint height = dimensions->y;

            if (gid.x >= width || gid.y >= height) {
                return;
            }

            float2 center = centers[0];
            float aspect_ratio = float(width) / float(height);
            float inv_zoom = 1.0 / zoom[0];

            float2 coord = float2(
                (float(gid.x) / float(width) - 0.5) * 2.0 * aspect_ratio * inv_zoom,
                (float(gid.y) / float(height) - 0.5) * 2.0 * inv_zoom
            ) + center;

            float2 z = float2(0.0);
            uint iter;
            float smooth_iter = 0.0;

            for (iter = 0; iter < max_iter[0]; iter++) {
                float x2 = z.x * z.x;
                float y2 = z.y * z.y;

                if (x2 + y2 > 4.0) {
                    float log_zn = log(x2 + y2) * 0.5;
                    float nu = log(log_zn / log(2.0)) / log(2.0);
                    smooth_iter = float(iter) + 1.0 - nu;
                    break;
                }
                z.y = 2.0 * z.x * z.y + coord.y;
                z.x = x2 - y2 + coord.x;
            }

            if (iter == max_iter[0]) {
                smooth_iter = float(max_iter[0]);
            }

            iterations[gid.y * width + gid.x] = smooth_iter;
        }
        "#;

        let compile_options = CompileOptions::new();

        let library = device.new_library_with_source(shader_source, &compile_options)
            .map_err(|e| anyhow::anyhow!("无法编译Metal着色器: {}", e))?;
        println!("✅ Metal着色器编译成功");

        let function = library.get_function("mandelbrot", None)
            .map_err(|e| anyhow::anyhow!("无法找到曼德勃罗集计算函数: {}", e))?;
        println!("✅ 成功获取计算函数");

        let pipeline_state = device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("无法创建计算管道: {}", e))?;
        println!("✅ 计算管道创建成功");

        Ok((device, pipeline_state))
    })
}

// 生成单帧图像 - 优化的并行/串行处理逻辑
fn generate_frame(
    frame_num: usize,
    total_frames: usize,
    config: &Config,
    _device: &Device,
    pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    centers_buffer: &Buffer,
    zoom_buffer: &Buffer,
    max_iter_buffer: &Buffer,
    dimensions_buffer: &Buffer,
    iterations_buffer: &Buffer,
) -> Result<(ImageBuffer<Rgb<u8>, Vec<u8>>, f64, usize, std::time::Duration)> {
    autoreleasepool(|| {
        let frame_ratio = frame_num as f64 / total_frames as f64;
        let zoom = config.start_zoom * config.zoom_rate.powf(config.duration * frame_ratio);
        let current_max_iter = (config.max_iterations as f64 *
            config.iteration_growth.powf(config.duration * frame_ratio)) as usize;

        // 更新缓冲区数据
        unsafe {
            centers_buffer.contents().copy_from(
                &[config.center_x as f32, config.center_y as f32] as *const _ as *const c_void,
                std::mem::size_of::<[f32; 2]>()
            );
        }

        unsafe {
            zoom_buffer.contents().copy_from(
                &[zoom as f32] as *const _ as *const c_void,
                std::mem::size_of::<f32>()
            );
        }

        unsafe {
            max_iter_buffer.contents().copy_from(
                &[current_max_iter as u32] as *const _ as *const c_void,
                std::mem::size_of::<u32>()
            );
        }

        let dimensions = [config.width as u32, config.height as u32];
        unsafe {
            dimensions_buffer.contents().copy_from(
                &dimensions as *const _ as *const c_void,
                std::mem::size_of::<[u32; 2]>()
            );
        }

        // 执行GPU计算
        let start_time = Instant::now();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(centers_buffer), 0);
        encoder.set_buffer(1, Some(zoom_buffer), 0);
        encoder.set_buffer(2, Some(max_iter_buffer), 0);
        encoder.set_buffer(3, Some(dimensions_buffer), 0);
        encoder.set_buffer(4, Some(iterations_buffer), 0);

        let thread_group_size = MTLSize::new(32, 32, 1);
        let tg_width = thread_group_size.width as usize;
        let tg_height = thread_group_size.height as usize;

        let thread_groups = MTLSize::new(
            ((config.width + tg_width - 1) / tg_width) as u64,
            ((config.height + tg_height - 1) / tg_height) as u64,
            1,
        );

        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let gpu_time = start_time.elapsed();

        // 读取迭代结果
        let iterations_count = config.width * config.height;
        let mut iterations = vec![0.0f32; iterations_count];
        let iterations_ptr = iterations_buffer.contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                iterations_ptr,
                iterations.as_mut_ptr(),
                iterations_count,
            );
        }

        // 创建图像缓冲区
        let mut image = ImageBuffer::new(config.width as u32, config.height as u32);
        let max_iter = current_max_iter as f32;
        let color_smoothing = config.color_smoothing;
        let palette = config.color_palette.as_str();
        let total_pixels = config.width * config.height;

        // 决定使用并行还是串行处理
        let use_parallel = config.parallel_color_mapping
            && total_pixels >= config.parallel_threshold_pixels;

        // 根据图像大小和配置选择最佳处理方式
        if use_parallel {
            // 并行处理 - 适合大图像
            image.enumerate_pixels_mut()
                .par_bridge()
                .for_each(|(x, y, pixel)| {
                    process_pixel(x, y, pixel, &iterations, config.width,
                                  max_iter, color_smoothing, palette);
                });
        } else {
            // 串行处理 - 适合小图像，避免线程开销
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                process_pixel(x, y, pixel, &iterations, config.width,
                              max_iter, color_smoothing, palette);
            }
        }

        Ok((image, zoom, current_max_iter, gpu_time))
    })
}

// 提取像素处理逻辑为单独函数 - 提高缓存利用率
#[inline(always)] // 强制内联，减少函数调用开销
fn process_pixel(
    x: u32,
    y: u32,
    pixel: &mut Rgb<u8>,
    iterations: &[f32],
    width: usize,
    max_iter: f32,
    color_smoothing: f64,
    palette: &str
) {
    let idx = (y as usize) * width + (x as usize);
    let iter = iterations[idx];

    let t = (iter / max_iter) as f64;
    let smoothed_t = t.powf(color_smoothing);

    let color = match palette {
        "classic" => classic_palette(smoothed_t),
        "fire" => fire_palette(smoothed_t),
        _ => smooth_palette(smoothed_t),
    };

    *pixel = Rgb(color);
}

// 色彩方案实现
fn classic_palette(t: f64) -> [u8; 3] {
    if t >= 1.0 {
        return [0, 0, 0];
    }
    let hue = (t * 0.7) % 1.0;
    hsv_to_rgb(hue as f32, 0.8, 0.9)
}

fn fire_palette(t: f64) -> [u8; 3] {
    if t >= 1.0 {
        return [0, 0, 0];
    }
    let t = t * 1.2;
    let r = (t.min(1.0) * 255.0) as u8;
    let g = ((t - 0.3).max(0.0).min(0.7) * 255.0 / 0.7) as u8;
    let b = ((t - 0.7).max(0.0).min(0.3) * 255.0 / 0.3) as u8;
    [r, g, b]
}

fn smooth_palette(t: f64) -> [u8; 3] {
    if t >= 1.0 {
        return [0, 0, 0];
    }
    let hue = 0.6 - t * 0.6;
    hsv_to_rgb(hue as f32, 0.9, 0.9)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h * 6.0 {
        h if h < 1.0 => (c, x, 0.0),
        h if h < 2.0 => (x, c, 0.0),
        h if h < 3.0 => (0.0, c, x),
        h if h < 4.0 => (0.0, x, c),
        h if h < 5.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [
        ((r + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b + m) * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

// 视频编码
fn encode_video(config: &Config) -> Result<()> {
    let frame_pattern = format!("{}/frame_%06d.bmp", config.temp_frame_dir);

    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(config.fps.to_string())
        .arg("-i")
        .arg(frame_pattern)
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("slow")
        .arg("-crf")
        .arg("16")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&config.output_file)
        .output()
        .with_context(|| "无法启动ffmpeg，请确保ffmpeg已安装并在PATH中")?;

    if !output.status.success() {
        let err_output = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg编码失败，错误信息: {}", err_output);
    }

    Ok(())
}

fn main() -> Result<()> {
    // 加载配置
    let config = load_config("config.toml")?;
    println!("使用配置: {:?}", config);

    // 计算总帧数
    let total_frames = (config.duration * config.fps as f64) as usize;
    println!(
        "视频时长: {:.1}秒, 帧率: {}fps, 总帧数: {}",
        config.duration, config.fps, total_frames
    );

    // 显示并行处理配置
    let total_pixels = config.width * config.height;
    println!(
        "颜色映射配置: 并行={}, 阈值像素={}, 当前图像像素={}, 将使用{}处理",
        config.parallel_color_mapping,
        config.parallel_threshold_pixels,
        total_pixels,
        if config.parallel_color_mapping && total_pixels >= config.parallel_threshold_pixels {
            "并行"
        } else {
            "串行"
        }
    );

    // 创建临时帧目录
    let temp_dir = &config.temp_frame_dir;
    fs::create_dir_all(temp_dir)
        .with_context(|| format!("无法创建临时目录: {}", temp_dir))?;

    // 初始化Metal
    let (device, pipeline) = init_metal()?;
    let command_queue = device.new_command_queue();
    println!("Metal初始化成功，开始GPU加速计算");

    // 预创建缓冲区
    let centers_buffer = device.new_buffer(
        std::mem::size_of::<[f32; 2]>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let zoom_buffer = device.new_buffer(
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let max_iter_buffer = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let dimensions_buffer = device.new_buffer(
        std::mem::size_of::<[u32; 2]>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let pixel_count = config.width * config.height;
    let iterations_buffer = device.new_buffer(
        (pixel_count * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // 创建进度条样式
    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} frames | 缩放: {zoom:.2} | 迭代: {iter} | GPU耗时: {gpu_time}"
    )?
        .progress_chars("#>-")
        .with_key("zoom", |_: &ProgressState, w: &mut dyn fmt::Write| write!(w, "").unwrap())
        .with_key("iter", |_: &ProgressState, w: &mut dyn fmt::Write| write!(w, "").unwrap())
        .with_key("gpu_time", |_: &ProgressState, w: &mut dyn fmt::Write| write!(w, "").unwrap());

    let pb = ProgressBar::new(total_frames as u64);
    pb.set_style(style);

    // 初始化异步保存通道
    let (sender, receiver) = mpsc::channel::<(ImageBuffer<Rgb<u8>, Vec<u8>>, String)>();

    // 启动保存线程
    thread::spawn(move || {
        while let Ok((image, path)) = receiver.recv() {
            if let Err(e) = image.save(&path) {
                eprintln!("保存图像失败: {} (路径: {})", e, path);
            }
        }
    });

    // 生成所有帧
    for frame in 0..total_frames {
        let (image, zoom, current_max_iter, gpu_time) = generate_frame(
            frame,
            total_frames,
            &config,
            &device,
            &pipeline,
            &command_queue,
            &centers_buffer,
            &zoom_buffer,
            &max_iter_buffer,
            &dimensions_buffer,
            &iterations_buffer,
        ).with_context(|| format!("生成帧 {} 失败", frame))?;

        // 更新进度条信息
        let current_zoom = zoom;
        let current_iter = current_max_iter;
        let current_gpu_time = gpu_time;

        pb.set_style(
            pb.style()
                .with_key("zoom", move |_: &ProgressState, w: &mut dyn fmt::Write| {
                    write!(w, "{:.2}", current_zoom).unwrap()
                })
                .with_key("iter", move |_: &ProgressState, w: &mut dyn fmt::Write| {
                    write!(w, "{}", current_iter).unwrap()
                })
                .with_key("gpu_time", move |_: &ProgressState, w: &mut dyn fmt::Write| {
                    write!(w, "{:?}", current_gpu_time).unwrap()
                })
        );

        pb.inc(1);

        // 发送到后台线程保存
        let frame_path = format!("{}/frame_{:06}.bmp", temp_dir, frame);
        sender.send((image, frame_path))
            .with_context(|| format!("发送帧 {} 到保存线程失败", frame))?;
    }

    // 完成处理
    drop(sender);
    pb.finish_with_message("所有帧生成完成，等待最后图像保存...");

    // 编码视频
    println!("正在使用ffmpeg编码视频...");
    encode_video(&config)?;

    // 清理临时文件
    fs::remove_dir_all(temp_dir)
        .with_context(|| format!("无法删除临时目录: {}", temp_dir))?;
    println!("临时文件已清理");

    println!("视频生成完成: {}", config.output_file);

    Ok(())
}
