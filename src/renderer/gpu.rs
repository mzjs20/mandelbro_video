use wgpu::*;
use crate::config::ColorScheme;
use crate::zoom::ZoomState;
use crate::renderer::{Renderer, generate_color_lut};
use anyhow::{Result, Context};
use std::sync::Arc;
use futures::executor::block_on;

/// GPU渲染器
pub struct GpuRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    params_buffer: Buffer,
    output_buffer: Buffer,
    color_lut_buffer: Buffer,
    bind_group: BindGroup,
    width: u32,
    height: u32,
    color_scheme: ColorScheme,
    supports_f64: bool,
}

impl GpuRenderer {
    pub fn new(width: u32, height: u32, color_scheme: ColorScheme) -> Result<Self> {
        block_on(async {
            Self::new_async(width, height, color_scheme).await
        })
    }

    async fn new_async(width: u32, height: u32, color_scheme: ColorScheme) -> Result<Self> {
        // 初始化wgpu
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("无法找到合适的GPU适配器")?;

        // 检查f64支持
        let features = adapter.features();
        let supports_f64 = features.contains(Features::SHADER_F64);

        log::info!("GPU适配器: {}", adapter.get_info().name);
        log::info!("支持f64: {}", supports_f64);

        let required_features = if supports_f64 {
            Features::SHADER_F64
        } else {
            Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Mandelbrot Device"),
                    required_features,
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .context("无法创建GPU设备")?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // 创建计算管线
        let shader_source = if supports_f64 {
            include_str!("mandelbrot_f64.wgsl")
        } else {
            include_str!("mandelbrot_f32.wgsl")
        };

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Mandelbrot Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Mandelbrot Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // 创建缓冲区
        let params_size = if supports_f64 { 64u64 } else { 32u64 };
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Params Buffer"),
            size: params_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_size = (width * height * 4) as u64;
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 颜色查找表缓冲区
        let max_lut_size = 10001u32; // 支持最大10000次迭代
        let color_lut_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Color LUT Buffer"),
            size: (max_lut_size * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建绑定组
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Mandelbrot Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: color_lut_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            params_buffer,
            output_buffer,
            color_lut_buffer,
            bind_group,
            width,
            height,
            color_scheme,
            supports_f64,
        })
    }

    /// 检查GPU是否支持f64
    pub fn check_f64_support() -> Result<bool> {
        block_on(async {
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .context("无法找到GPU适配器")?;

            Ok(adapter.features().contains(Features::SHADER_F64))
        })
    }

    fn update_params(&self, state: &ZoomState) {
        // 计算视图范围
        let aspect_ratio = self.width as f64 / self.height as f64;
        let view_size = 4.0 / state.zoom;
        let half_width = view_size / 2.0;
        let half_height = half_width / aspect_ratio;

        let x_min = state.center_re - half_width;
        let y_min = state.center_im - half_height;
        let x_range = half_width * 2.0;
        let y_range = half_height * 2.0;

        if self.supports_f64 {
            // f64参数
            let mut params = [0u8; 64];
            params[0..8].copy_from_slice(&x_min.to_le_bytes());
            params[8..16].copy_from_slice(&y_min.to_le_bytes());
            params[16..24].copy_from_slice(&x_range.to_le_bytes());
            params[24..32].copy_from_slice(&y_range.to_le_bytes());
            params[32..36].copy_from_slice(&state.max_iter.to_le_bytes());
            params[36..40].copy_from_slice(&self.width.to_le_bytes());
            params[40..44].copy_from_slice(&self.height.to_le_bytes());
            self.queue.write_buffer(&self.params_buffer, 0, &params);
        } else {
            // f32参数
            let mut params = [0u8; 32];
            params[0..4].copy_from_slice(&(x_min as f32).to_le_bytes());
            params[4..8].copy_from_slice(&(y_min as f32).to_le_bytes());
            params[8..12].copy_from_slice(&(x_range as f32).to_le_bytes());
            params[12..16].copy_from_slice(&(y_range as f32).to_le_bytes());
            params[16..20].copy_from_slice(&state.max_iter.to_le_bytes());
            params[20..24].copy_from_slice(&self.width.to_le_bytes());
            params[24..28].copy_from_slice(&self.height.to_le_bytes());
            self.queue.write_buffer(&self.params_buffer, 0, &params);
        }

        // 更新颜色查找表 (每个颜色用u32存储，小端序: R G B 0)
        let lut = generate_color_lut(state.max_iter.min(10000), self.color_scheme);
        let lut_bytes: Vec<u8> = lut.iter()
            .map(|c| {
                // 小端序: R G B 0 (这样读取时 chunk[0]=R, chunk[1]=G, chunk[2]=B)
                let color: u32 = (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16);
                color
            })
            .flat_map(|c| c.to_le_bytes())
            .collect();
        self.queue.write_buffer(&self.color_lut_buffer, 0, &lut_bytes);
    }
}

impl Renderer for GpuRenderer {
    fn render(&mut self, state: &ZoomState) -> Result<Vec<u8>> {
        self.update_params(state);

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Mandelbrot Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Mandelbrot Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(
                (self.width + 15) / 16,
                (self.height + 15) / 16,
                1,
            );
        }

        // 复制输出到可读缓冲区
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.width * self.height * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging_buffer,
            0,
            (self.width * self.height * 4) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // 读取结果
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(Maintain::Wait);

        rx.recv()??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u8> = data.iter().copied().collect();
        drop(data);

        // 转换RGBA到RGB
        let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
        for chunk in result.chunks(4) {
            rgb.push(chunk[0]);
            rgb.push(chunk[1]);
            rgb.push(chunk[2]);
        }

        Ok(rgb)
    }

    fn name(&self) -> &'static str {
        if self.supports_f64 {
            "GPU (f64)"
        } else {
            "GPU (f32)"
        }
    }
}