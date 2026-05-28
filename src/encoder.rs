use std::process::{Child, ChildStdin, Command, Stdio};
use std::io::Write;
use anyhow::{Context, Result};
use crate::config::EncodingConfig;

pub struct Av1Encoder {
    writer: Option<BufWriter<ChildStdin>>,
    ffmpeg: Option<Child>,
    frame_size: usize,
    encoded_frames: u32,
}

use std::io::BufWriter;

impl Av1Encoder {
    pub fn start(
        width: u32,
        height: u32,
        fps: u32,
        total_frames: u32,
        output: String,
        encoding: EncodingConfig,
    ) -> Result<Self> {
        log::info!("开始编码视频: {}x{} @ {}fps", width, height, fps);
        log::info!("总帧数: {}, 输出文件: {}", total_frames, output);

        let frame_size = (width as usize) * (height as usize) * 3;
        log::info!("每帧大小: {} 字节", frame_size);

        let (codec, encoder_args) = match output.as_str() {
            s if s.ends_with(".mp4") => {
                let args = format!("-crf {} -preset medium -pix_fmt yuv420p -tune stillimage -x264-params mbtree=0:weightb=0", encoding.quantizer);
                ("libx264", args)
            }
            s if s.ends_with(".webm") => {
                let args = format!("-crf {} -b:v 0 -pix_fmt yuv420p", encoding.quantizer);
                ("libvpx-vp9", args)
            }
            s if s.ends_with(".mkv") => {
                let args = format!("-crf {} -preset medium -pix_fmt yuv420p -tune stillimage", encoding.quantizer);
                ("libx264", args)
            }
            s if s.ends_with(".ivf") => {
                let args = format!("-crf {} -cpu-used 6 -row-mt 1 -pix_fmt yuv420p", encoding.quantizer);
                ("libaom-av1", args)
            }
            _ => {
                let args = format!("-crf {} -preset medium -pix_fmt yuv420p", encoding.quantizer);
                ("libx264", args)
            }
        };

        log::info!("使用编码器: {}", codec);

        let ffmpeg_cmd = format!(
            "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s {}x{} -r {} -i - -c:v {} {} {}",
            width, height, fps, codec, encoder_args, output
        );
        log::info!("FFmpeg命令: {}", ffmpeg_cmd);

        let mut ffmpeg = Command::new("ffmpeg")
            .arg("-y")
            .arg("-f").arg("rawvideo")
            .arg("-pix_fmt").arg("rgb24")
            .arg("-s").arg(format!("{}x{}", width, height))
            .arg("-r").arg(fps.to_string())
            .arg("-i").arg("-")
            .arg("-c:v").arg(codec)
            .args(encoder_args.split_whitespace())
            .arg(output)
            .stdin(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("启动FFmpeg失败，请确保已安装ffmpeg")?;

        let writer = BufWriter::new(
            ffmpeg.stdin.take().context("无法获取FFmpeg stdin")?
        );

        Ok(Self {
            writer: Some(writer),
            ffmpeg: Some(ffmpeg),
            frame_size,
            encoded_frames: 0,
        })
    }

    pub fn push_frame(&mut self, frame_data: &[u8]) -> Result<()> {
        if frame_data.len() != self.frame_size {
            anyhow::bail!("帧大小不匹配: 期望 {}, 实际 {}", self.frame_size, frame_data.len());
        }
        if let Some(ref mut writer) = self.writer {
            writer.write_all(frame_data)
                .context("写入FFmpeg失败")?;
        }
        self.encoded_frames += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        // 关闭 stdin，让 FFmpeg 完成编码
        drop(self.writer.take());
        if let Some(mut ffmpeg) = self.ffmpeg.take() {
            let status = ffmpeg.wait()
                .context("等待FFmpeg完成失败")?;
            if !status.success() {
                anyhow::bail!("FFmpeg退出码: {}", status);
            }
        }
        log::info!("视频已保存 ({}帧)", self.encoded_frames);
        Ok(())
    }
}
