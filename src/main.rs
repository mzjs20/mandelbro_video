use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::fs;
use std::process::Command;
use std::time::Instant;
use wgpu::util::DeviceExt;

#[derive(Debug, Deserialize)]
struct Config {
    rendering: RenderingConfig,
    mandelbrot: MandelbrotConfig,
}

#[derive(Debug, Deserialize)]
struct RenderingConfig {
    width: u32,
    height: u32,
    fps: u32,
    total_frames: u32,
}

#[derive(Debug, Deserialize)]
struct MandelbrotConfig {
    center_x: f64,
    center_y: f64,
    initial_radius: f64,
    zoom_factor: f64,
    initial_iterations: u32,
    max_iterations: u32,
    iteration_growth_factor: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ShaderParams {
    width: f32,
    height: f32,
    center_x: f32,
    center_y: f32,
    zoom: f32,
    max_iterations: u32,
    fast_escape_threshold: f32,
    use_f64: u32,
}
unsafe impl Pod for ShaderParams {}
unsafe impl Zeroable for ShaderParams {}

impl ShaderParams {
    fn new(
        width: u32,
        height: u32,
        center_x: f64,
        center_y: f64,
        zoom: f64,
        max_iterations: u32,
        use_f64: bool
    ) -> Self {
        Self {
            width: width as f32,
            height: height as f32,
            center_x: center_x as f32,
            center_y: center_y as f32,
            zoom: zoom as f32,
            max_iterations,
            fast_escape_threshold: 4.0,
            use_f64: use_f64 as u32,
        }
    }
}

const WORKGROUP_SIZE: u32 = 16;

// 新增：控制是否强制使用f64的选项（对于不支持f64的设备会失败）
const FORCE_F64: bool = false;

fn init_wgpu() -> Result<(wgpu::Device, wgpu::Queue, wgpu::Adapter, bool)> {
    pollster::block_on(async {
        let instance_descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
        };

        let instance = wgpu::Instance::new(&instance_descriptor);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("无法获取GPU适配器")?;

        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();

        // 检查是否支持f64
        let supports_f64_hw = adapter_features.contains(wgpu::Features::SHADER_F64);
        // 根据硬件支持和强制选项决定是否使用f64
        let use_f64 = FORCE_F64 || supports_f64_hw;

        if FORCE_F64 {
            println!("强制启用f64精度模式...");
            if !supports_f64_hw {
                eprintln!("警告：当前GPU硬件不支持f64精度，强制启用可能导致失败");
            }
        } else if !supports_f64_hw {
            eprintln!("当前GPU不支持f64精度，将使用f32模式运行");
        } else {
            println!("GPU支持f64精度，将使用高精度模式运行");
        }

        println!("使用适配器: {}", adapter_info.name);

        // 明确声明需要的特性
        let mut required_features = wgpu::Features::empty();
        if use_f64 {
            required_features.insert(wgpu::Features::SHADER_F64);
        }

        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Mandelbrot Device"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits()),
            memory_hints: Default::default(),
            trace: Default::default(),
        };

        // 尝试创建设备，处理可能的特性不支持错误
        let (device, queue) = match adapter.request_device(&device_descriptor).await {
            Ok(result) => result,
            Err(e) => {
                if use_f64 {
                    anyhow::bail!(
                        "无法创建支持f64的设备，当前GPU可能不支持该特性。错误: {}",
                        e
                    );
                } else {
                    anyhow::bail!("无法创建设备: {}", e);
                }
            }
        };

        Ok((device, queue, adapter, use_f64))
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    use_f64: bool
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    let (precision_type, param_type) = if use_f64 {
        ("f64", "f64")
    } else {
        ("f32", "f32")
    };

    // 根据精度类型定义平滑迭代计算的代码
    let smooth_iter_code = if use_f64 {
        // f64模式下使用高精度计算
        "let smooth_iter = f32(f64(iterations) + 1.0 - log_zn);"
    } else {
        // f32模式下使用单精度计算
        "let smooth_iter = f32(iterations) + 1.0 - log_zn;"
    };

    let shader_source = format!(
        r#"
struct ShaderParams {{
    width: f32,
    height: f32,
    center_x: {1},
    center_y: {1},
    zoom: {1},
    max_iterations: u32,
    fast_escape_threshold: f32,
    use_f64: u32,
}};

struct MandelbrotResult {{
    iterations: u32,
    z_re: {0},
    z_im: {0},
}};

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: ShaderParams;

fn mandelbrot(c_re: {0}, c_im: {0}, max_iter: u32) -> MandelbrotResult {{
    var z_re: {0} = 0.0;
    var z_im: {0} = 0.0;
    var iter: u32 = 0u;

    if c_re * c_re + c_im * c_im > {0}(params.fast_escape_threshold) {{
        return MandelbrotResult(0u, z_re, z_im);
    }}

    while iter < max_iter {{
        let z_re2 = z_re * z_re;
        let z_im2 = z_im * z_im;
        let z_sum = z_re2 + z_im2;

        if z_sum > {0}(4.0) {{
            break;
        }}

        if iter > 100u && z_sum > {0}(100.0) {{
            let log2_z_sum = log2(z_sum);
            iter += u32(log2_z_sum / {0}(2.0));
            if iter >= max_iter {{
                iter = max_iter - 1u;
            }}
            break;
        }}

        let new_re = z_re2 - z_im2 + c_re;
        z_im = {0}(2.0) * z_re * z_im + c_im;
        z_re = new_re;
        iter += 1u;
    }}

    return MandelbrotResult(iter, z_re, z_im);
}}

fn gamma_correct(value: f32) -> f32 {{
    if value <= 0.0031308 {{
        return value * 12.92;
    }} else {{
        return 1.055 * pow(value, 1.0 / 2.2) - 0.055;
    }}
}}

fn get_color(iterations: u32, max_iter: u32, z_re: {0}, z_im: {0}) -> u32 {{
    if iterations == max_iter {{
        return 0xFF000000u;
    }}

    let z_mag = sqrt(z_re * z_re + z_im * z_im);
    let log_zn = log(log(z_mag) / log({0}(2.0))) / log({0}(2.0));
    {2}
    let t = smooth_iter / f32(max_iter);

    var r: f32;
    var g: f32;
    var b: f32;

    if t < 0.16 {{
        let local_t = t / 0.16;
        r = 0.0;
        g = 0.0;
        b = 0.3 + 0.69 * local_t;
    }} else if t < 0.42 {{
        let local_t = (t - 0.16) / 0.26;
        r = 0.0;
        g = 0.5 * local_t;
        b = 1.0 - 0.45 * local_t;
    }} else if t < 0.6425 {{
        let local_t = (t - 0.42) / 0.2225;
        r = local_t;
        g = 0.5 + 0.5 * local_t;
        b = 0.5 * (1.0 - local_t);
    }} else if t < 0.8575 {{
        let local_t = (t - 0.6425) / 0.215;
        r = 1.0;
        g = 1.0 - 0.5 * local_t;
        b = 0.0;
    }} else if t < 0.93 {{
        let local_t = (t - 0.8575) / 0.0725;
        r = 1.0;
        g = 0.5 + 0.25 * local_t;
        b = 0.0 + 0.5 * local_t;
    }} else {{
        let local_t = (t - 0.93) / 0.07;
        r = 1.0 - 0.1 * local_t;
        g = 0.75 + 0.25 * local_t;
        b = 0.5 + 0.5 * local_t;
    }}

    r = gamma_correct(r) * 255.0;
    g = gamma_correct(g) * 255.0;
    b = gamma_correct(b) * 255.0;

    let red = clamp(u32(r), 0u, 255u);
    let green = clamp(u32(g), 0u, 255u);
    let blue = clamp(u32(b), 0u, 255u);

    return (0xFFu << 24) | (blue << 16) | (green << 8) | red;
}}

@compute @workgroup_size({3}, {3}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let width = params.width;
    let height = params.height;

    if (global_id.x >= u32(width) || global_id.y >= u32(height)) {{
        return;
    }}

    let aspect_ratio = height / width;

    let x = (f32(global_id.x) - width * 0.5) / (width * 0.5) * params.zoom;
    let y = (f32(global_id.y) - height * 0.5) / (height * 0.5) * params.zoom * aspect_ratio;

    let c_re = {0}(params.center_x) + {0}(x);
    let c_im = {0}(params.center_y) + {0}(y);

    let result = mandelbrot(c_re, c_im, params.max_iterations);
    let color = get_color(result.iterations, params.max_iterations, result.z_re, result.z_im);

    let index = global_id.y * u32(width) + global_id.x;
    output[index] = color;
}}
"#,
        precision_type,  // {0} - 计算精度类型
        param_type,      // {1} - 参数类型
        smooth_iter_code,// {2} - 平滑迭代计算代码
        WORKGROUP_SIZE   // {3} - 工作组大小
    );

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mandelbrot Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mandelbrot Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mandelbrot Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mandelbrot Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok((compute_pipeline, bind_group_layout))
}

fn generate_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    compute_pipeline: &wgpu::ComputePipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    config: &Config,
    zoom: f64,
    current_iterations: u32,
    use_f64: bool,
) -> Result<(Vec<u8>, std::time::Duration)> {
    pollster::block_on(async {
        let width = config.rendering.width;
        let height = config.rendering.height;

        let params = ShaderParams::new(
            width,
            height,
            config.mandelbrot.center_x,
            config.mandelbrot.center_y,
            zoom,
            current_iterations,
            use_f64,
        );

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_size = (width * height * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mandelbrot Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_start = Instant::now();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mandelbrot Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mandelbrot Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                1,
            );
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit([encoder.finish()]);

        let compute_time = compute_start.elapsed();

        let (sender, receiver) = std::sync::mpsc::channel();
        staging_buffer.map_async(
            wgpu::MapMode::Read,
            0..output_size,
            move |result| {
                sender.send(result).unwrap();
            },
        );

        let _ = device.poll(wgpu::PollType::Wait);
        receiver.recv().context("缓冲区映射被中断")??;

        let mapping = staging_buffer.get_mapped_range(0..output_size);
        let result = mapping.to_vec();

        drop(mapping);
        staging_buffer.unmap();

        Ok((result, compute_time))
    })
}

fn save_frame(
    frame_data: &[u8],
    temp_dir: &str,
    frame: u32,
    config: &Config,
) -> Result<()> {
    let width = config.rendering.width;
    let height = config.rendering.height;

    let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

    for i in 0..(width * height) as usize {
        let offset = i * 4;
        if offset + 3 < frame_data.len() {
            rgb_data.push(frame_data[offset]);     // B
            rgb_data.push(frame_data[offset + 1]); // G
            rgb_data.push(frame_data[offset + 2]); // R
        }
    }

    let image: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_data)
        .context("无法从数据创建图像")?;

    let filename = format!("{}/frame_{:06}.bmp", temp_dir, frame);
    image.save(&filename).context("无法保存帧图像")?;

    Ok(())
}

fn create_video(temp_dir: &str, config: &Config) -> Result<()> {
    let output_file = "mandelbrot_zoom.mp4";

    println!("正在生成视频...");

    let status = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-framerate",
            &config.rendering.fps.to_string(),
            "-i",
            &format!("{}/frame_%06d.bmp", temp_dir),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "medium",
            output_file,
        ])
        .status()
        .context("无法执行FFmpeg命令")?;

    if !status.success() {
        anyhow::bail!("FFmpeg命令执行失败");
    }

    Ok(())
}

fn format_duration(d: &std::time::Duration) -> String {
    let micros = d.as_micros();
    if micros < 1000 {
        format!("{}µs", micros)
    } else if micros < 1_000_000 {
        format!("{:.2}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}

fn main() -> Result<()> {
    let config_str = fs::read_to_string("config.toml").context("无法读取配置文件")?;
    let config: Config = toml::from_str(&config_str).context("配置文件格式错误")?;

    println!("开始生成曼德勃罗集视频...");
    println!("分辨率: {}x{}", config.rendering.width, config.rendering.height);
    println!("总帧数: {}", config.rendering.total_frames);
    println!("帧率: {} FPS", config.rendering.fps);
    println!("初始迭代次数: {}", config.mandelbrot.initial_iterations);
    println!("最大迭代次数: {}", config.mandelbrot.max_iterations);
    println!("迭代增长因子: {:.6}", config.mandelbrot.iteration_growth_factor);
    println!();

    let temp_dir = "temp_frames";
    if std::path::Path::new(temp_dir).exists() {
        fs::remove_dir_all(temp_dir).context("无法清理临时目录")?;
    }
    fs::create_dir_all(temp_dir).context("无法创建临时目录")?;

    println!("初始化wgpu...");
    let (device, queue, adapter, use_f64) = init_wgpu()?;
    println!("使用GPU: {}", adapter.get_info().name);

    println!("创建计算管线...");
    let (compute_pipeline, bind_group_layout) = create_compute_pipeline(&device, use_f64)?;

    let progress_bar = ProgressBar::new(config.rendering.total_frames as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta} | {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    println!("开始生成帧...");

    let mut frame_times = Vec::new();
    let total_start_time = Instant::now();

    for frame in 0..config.rendering.total_frames {
        let zoom = config.mandelbrot.initial_radius *
            config.mandelbrot.zoom_factor.powi(-(frame as i32));
        let zoom_factor = config.mandelbrot.zoom_factor.powi(frame as i32);

        let current_iterations = if zoom_factor < 1000.0 {
            let growth = config.mandelbrot.iteration_growth_factor.powi(frame as i32);
            ((config.mandelbrot.initial_iterations as f64 * growth) as u32)
                .min(config.mandelbrot.max_iterations)
        } else if zoom_factor < 1e6 {
            let growth = config.mandelbrot.iteration_growth_factor.powi((frame as f64 * 1.3) as i32);
            ((config.mandelbrot.initial_iterations as f64 * growth * 1.5) as u32)
                .min(config.mandelbrot.max_iterations)
        } else {
            let growth = config.mandelbrot.iteration_growth_factor.powi((frame as f64 * 1.5) as i32);
            ((config.mandelbrot.initial_iterations as f64 * growth * 2.0) as u32)
                .min(config.mandelbrot.max_iterations)
        };

        let (frame_data, compute_time) = generate_frame(
            &device,
            &queue,
            &compute_pipeline,
            &bind_group_layout,
            &config,
            zoom,
            current_iterations,
            use_f64,
        )?;

        let save_start = Instant::now();
        save_frame(&frame_data, temp_dir, frame, &config)?;
        let save_time = save_start.elapsed();

        let total_frame_time = compute_time + save_time;
        frame_times.push(total_frame_time);

        progress_bar.set_message(format!(
            "帧 {} | 缩放: {:.2e}x | 迭代: {} | 精度: {} | GPU: {} | 保存: {}",
            frame + 1,
            zoom_factor,
            current_iterations,
            if use_f64 { "f64" } else { "f32" },
            format_duration(&compute_time),
            format_duration(&save_time),
        ));

        progress_bar.inc(1);
    }

    let total_time = total_start_time.elapsed();

    progress_bar.finish_with_message("所有帧生成完成");

    let total_compute_time: std::time::Duration = frame_times.iter().sum();
    let avg_frame_time = total_compute_time / config.rendering.total_frames as u32;
    let min_frame_time = frame_times.iter().min().unwrap();
    let max_frame_time = frame_times.iter().max().unwrap();

    println!();
    println!("帧生成统计:");
    println!("  总耗时: {:.2}s", total_time.as_secs_f64());
    println!("  平均每帧: {}", format_duration(&avg_frame_time));
    println!("  最快帧: {}", format_duration(min_frame_time));
    println!("  最慢帧: {}", format_duration(max_frame_time));
    println!("  最终缩放: {:.2e}x", config.mandelbrot.zoom_factor.powi(config.rendering.total_frames as i32));
    println!();

    let video_start = Instant::now();
    create_video(temp_dir, &config)?;
    let video_time = video_start.elapsed();

    println!("视频编码耗时: {:.2}s", video_time.as_secs_f64());

    fs::remove_dir_all(temp_dir).context("无法清理临时目录")?;

    println!("视频生成完成: mandelbrot_zoom.mp4");
    Ok(())
}
