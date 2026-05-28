/// GPU Shader for Mandelbrot (f32 precision)

struct Params {
    x_min: f32,
    y_min: f32,
    x_range: f32,
    y_range: f32,
    max_iter: u32,
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> color_lut: array<u32>;

// 主心形检测
fn in_main_cardioid(c_re: f32, c_im: f32) -> bool {
    let q = (c_re - 0.25) * (c_re - 0.25) + c_im * c_im;
    return q * (q + (c_re - 0.25)) <= 0.25 * c_im * c_im;
}

// 周期2圆盘检测
fn in_period2_bulb(c_re: f32, c_im: f32) -> bool {
    return (c_re + 1.0) * (c_re + 1.0) + c_im * c_im <= 0.0625;
}

fn mandelbrot(c_re: f32, c_im: f32, max_iter: u32) -> u32 {
    // 快速检测
    if in_main_cardioid(c_re, c_im) || in_period2_bulb(c_re, c_im) {
        return max_iter;
    }

    var z_re: f32 = 0.0;
    var z_im: f32 = 0.0;
    var z_re_old: f32 = 0.0;
    var z_im_old: f32 = 0.0;
    var period: u32 = 0u;

    for (var i: u32 = 0u; i < max_iter; i = i + 1u) {
        let z_re_sq = z_re * z_re;
        let z_im_sq = z_im * z_im;

        if (z_re_sq + z_im_sq > 4.0) {
            return i;
        }

        let z_im_new = 2.0 * z_re * z_im + c_im;
        z_re = z_re_sq - z_im_sq + c_re;
        z_im = z_im_new;

        // 周期性检测
        if (z_re == z_re_old && z_im == z_im_old) {
            return max_iter;
        }

        period = period + 1u;
        if (period > 20u) {
            period = 0u;
            z_re_old = z_re;
            z_im_old = z_im;
        }
    }

    return max_iter;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let c_re = params.x_min + f32(x) / f32(params.width) * params.x_range;
    let c_im = params.y_min + f32(y) / f32(params.height) * params.y_range;

    let iter = mandelbrot(c_re, c_im, params.max_iter);

    // 从颜色查找表获取颜色 (每个u32存储一个RGBA颜色)
    let color = color_lut[iter];
    let idx = y * params.width + x;
    output[idx] = color;
}