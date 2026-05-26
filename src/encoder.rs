use crate::config::EncodingConfig;
use anyhow::{Result, Context};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{Write, BufWriter};
use std::process::{Command, Stdio};
use std::thread;

/// AV1视频编码器（使用FFmpeg管道方式）
pub struct Av1Encoder {
    config: EncodingConfig,
    width: u32,
    height: u32,
    fps: u32,
    output_path: String,
    progress_bar: ProgressBar,
}

impl Av1Encoder {
    pub fn new(
        width: u32,
        height: u32,
        fps: u32,
        total_frames: u32,
        output_path: String,
        config: EncodingConfig,
    ) -> Result<Self> {
        let progress_bar = ProgressBar::new(total_frames as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.green/blue} {pos}/{len} 帧 ({eta}) 编码中")
                .unwrap()
                .progress_chars("█▓░"),
        );

        Ok(Self {
            config,
            width,
            height,
            fps,
            output_path,
            progress_bar,
        })
    }

    /// 编码视频 - 使用FFmpeg管道
    pub fn encode(&mut self, frames: Vec<Vec<u8>>) -> Result<()> {
        log::info!("开始编码视频: {}x{} @ {}fps", self.width, self.height, self.fps);
        log::info!("总帧数: {}, 输出文件: {}", frames.len(), self.output_path);

        let frame_size = (self.width * self.height * 3) as usize;
        log::info!("每帧大小: {} 字节", frame_size);

        // 根据输出文件扩展名选择编码器
        let extension = self.output_path.rsplit('.').next().unwrap_or("mp4");
        let (codec, extra_args) = self.get_codec_settings(extension);

        log::info!("使用编码器: {}", codec);

        // 构建FFmpeg命令
        let mut args = vec![
            "-y".to_string(),
            "-f".to_string(), "rawvideo".to_string(),
            "-pix_fmt".to_string(), "rgb24".to_string(),
            "-s".to_string(), format!("{}x{}", self.width, self.height),
            "-r".to_string(), self.fps.to_string(),
            "-i".to_string(), "-".to_string(),
            "-c:v".to_string(), codec,
        ];

        args.extend(extra_args);
        args.push(self.output_path.clone());

        log::info!("FFmpeg命令: ffmpeg {}", args.join(" "));

        // 启动FFmpeg进程 - 使用stderr管道来读取输出
        let mut ffmpeg = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("无法启动FFmpeg进程，请确保已安装FFmpeg")?;

        let stdin = ffmpeg.stdin.take().context("无法获取FFmpeg stdin")?;
        let mut stderr = ffmpeg.stderr.take().context("无法获取FFmpeg stderr")?;

        // 在后台线程中读取stderr，避免死锁
        let stderr_thread = thread::spawn(move || {
            let mut err_output = String::new();
            std::io::Read::read_to_string(&mut stderr, &mut err_output).ok();
            err_output
        });

        // 写入所有帧
        let mut writer = BufWriter::with_capacity(1024 * 1024, stdin);
        for (idx, frame_data) in frames.iter().enumerate() {
            if frame_data.len() != frame_size {
                log::warn!("帧 {} 大小不匹配: 期望 {}, 实际 {}", idx, frame_size, frame_data.len());
            }
            writer.write_all(frame_data)
                .with_context(|| format!("写入帧 {} 数据失败", idx))?;
            self.progress_bar.inc(1);

            // 每100帧刷新一次，避免缓冲区过大
            if idx % 100 == 0 {
                writer.flush().context("刷新FFmpeg输入失败")?;
            }
        }
        writer.flush().context("最终刷新FFmpeg输入失败")?;
        drop(writer); // 显式关闭stdin

        log::info!("所有帧已写入，等待FFmpeg完成...");

        // 等待FFmpeg完成
        let status = ffmpeg.wait().context("等待FFmpeg进程失败")?;

        // 获取stderr输出
        let err_output = stderr_thread.join().unwrap_or_default();

        if !status.success() {
            log::error!("FFmpeg stderr: {}", err_output);
            return Err(anyhow::anyhow!("FFmpeg编码失败 (退出码 {:?}): {}", status.code(),
                if err_output.is_empty() { "无错误信息" } else { &err_output }));
        }

        self.progress_bar.finish_with_message("编码完成!");
        log::info!("视频已保存至: {}", self.output_path);

        Ok(())
    }

    /// 根据文件扩展名和配置选择编码器设置
    fn get_codec_settings(&self, extension: &str) -> (String, Vec<String>) {
        match extension {
            "ivf" => {
                // IVF格式用于AV1
                let crf = self.config.quantizer;
                (
                    "libaom-av1".to_string(),
                    vec![
                        "-crf".to_string(), crf.to_string(),
                        "-b:v".to_string(), "0".to_string(),
                        "-cpu-used".to_string(), "4".to_string(), // 更快的预设
                        "-strict".to_string(), "experimental".to_string(),
                    ],
                )
            }
            "mp4" | "mkv" => {
                // 默认使用H.264，更快
                let crf = self.config.quantizer;
                (
                    "libx264".to_string(),
                    vec![
                        "-crf".to_string(), crf.to_string(),
                        "-preset".to_string(), "ultrafast".to_string(),
                        "-pix_fmt".to_string(), "yuv420p".to_string(),
                    ],
                )
            }
            "webm" => {
                // WebM使用VP9或AV1
                let crf = self.config.quantizer;
                (
                    "libvpx-vp9".to_string(),
                    vec![
                        "-crf".to_string(), crf.to_string(),
                        "-b:v".to_string(), "0".to_string(),
                        "-speed".to_string(), "4".to_string(),
                    ],
                )
            }
            _ => {
                // 默认H.264
                let crf = self.config.quantizer;
                (
                    "libx264".to_string(),
                    vec![
                        "-crf".to_string(), crf.to_string(),
                        "-preset".to_string(), "fast".to_string(),
                        "-pix_fmt".to_string(), "yuv420p".to_string(),
                    ],
                )
            }
        }
    }
}