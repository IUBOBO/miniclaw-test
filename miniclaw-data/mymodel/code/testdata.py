import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter


# ============================================================
# 全局设置
# ============================================================

# 固定随机种子：
# 这样每次运行生成的 Gaussian Noise / Stripe / Cloud 基本一致，
# 方便论文实验复现。
# 如果希望每次随机生成，可以删除这一行。
np.random.seed(42)


# ============================================================
# 1. Gaussian Blur
# ============================================================

def add_blur(image_float, w=5):
    """
    添加高斯模糊

    Parameters
    ----------
    image_float : ndarray
        输入图像，形状 (H, W, C)，float32，[0, 255]

    w : int
        模糊窗口强度。
        数值越大，模糊越严重。

        推荐：
        3   -> 轻微
        5   -> 中等
        7   -> 较严重
        9   -> 严重
        11  -> 很严重
    """

    sigma = (w - 1) / 6.0

    blurred = np.stack([
        gaussian_filter(
            image_float[:, :, i],
            sigma=sigma
        )
        for i in range(image_float.shape[2])
    ], axis=-1)

    return blurred


# ============================================================
# 2. Gaussian Noise
# ============================================================

def add_gaussian_noise(image_float, sigma=20.0):
    """
    添加高斯随机噪声

    sigma 越大，噪声越严重。

    推荐：
    5   -> 轻微
    10  -> 较轻
    20  -> 中等
    30  -> 较严重
    40  -> 严重
    50  -> 很严重
    """

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image_float.shape
    ).astype(np.float32)

    noisy = image_float + noise

    return noisy


# ============================================================
# 3. Stripe Noise
# ============================================================

def add_stripe_noise(
        image_float,
        r=0.2,
        probability=0.1,
        direction='horizontal',
        dtype_max=255.0
):
    """
    添加条带噪声

    Parameters
    ----------
    r : float
        条带亮度变化幅度。

        越大，每条条带越明显。

        推荐：
        0.05 -> 轻微
        0.10 -> 较轻
        0.20 -> 中等
        0.30 -> 较严重
        0.40 -> 严重
        0.50 -> 很严重

    probability : float
        每一行 / 列成为条带的概率。

        越大，条带数量越多。

        推荐：
        0.05 -> 稀疏
        0.10 -> 中等
        0.20 -> 较密集
        0.30 -> 非常密集

    direction :
        horizontal -> 水平条带
        vertical   -> 垂直条带
    """

    noisy = image_float.copy()

    h, w = image_float.shape[:2]

    max_offset = r * dtype_max

    if direction == 'horizontal':

        for i in range(h):

            if np.random.rand() < probability:

                offset = np.random.uniform(
                    -max_offset,
                    max_offset
                )

                noisy[i, :, :] += offset

    elif direction == 'vertical':

        for j in range(w):

            if np.random.rand() < probability:

                offset = np.random.uniform(
                    -max_offset,
                    max_offset
                )

                noisy[:, j, :] += offset

    else:

        raise ValueError(
            "direction 必须为 'horizontal' 或 'vertical'"
        )

    return noisy


# ============================================================
# 4. Fog
# ============================================================

def add_fog(
        image_float,
        A=0.9,
        t0=0.6,
        dtype_max=255.0
):
    """
    添加雾霾退化

    使用简单大气散射模型：

        I_fog = I * t + A * (1 - t)

    Parameters
    ----------
    A :
        大气光。

        越接近 1，雾越白。

    t0 :
        透射率，是控制雾强度最重要的参数。

        越小 -> 雾越严重

        推荐：
        0.85 -> 很轻
        0.75 -> 轻度
        0.60 -> 中度
        0.45 -> 较重
        0.30 -> 重度
        0.15 -> 非常严重
    """

    A_abs = A * dtype_max

    hazy = (
        image_float * t0
        +
        A_abs * (1.0 - t0)
    )

    return hazy


# ============================================================
# 5. Perlin Noise
# 用于 Cloud Occlusion
# ============================================================

def generate_perlin_noise_2d(shape, res):
    """
    生成二维 Perlin Noise。

    相比原来的代码，这里增加了尺寸适配，
    即使 H/W 不能被 res 整除，也不会直接报错。
    """

    original_h, original_w = shape

    res_y, res_x = res

    # --------------------------------------------------------
    # 为了保证尺寸能整除 res，先向上扩展
    # --------------------------------------------------------

    padded_h = int(
        np.ceil(original_h / res_y) * res_y
    )

    padded_w = int(
        np.ceil(original_w / res_x) * res_x
    )

    shape_pad = (
        padded_h,
        padded_w
    )

    # --------------------------------------------------------
    # Perlin 基础函数
    # --------------------------------------------------------

    def f(t):
        return (
            6 * t ** 5
            - 15 * t ** 4
            + 10 * t ** 3
        )

    delta = (
        res_y / shape_pad[0],
        res_x / shape_pad[1]
    )

    d = (
        shape_pad[0] // res_y,
        shape_pad[1] // res_x
    )

    grid = np.mgrid[
        0:res_y:delta[0],
        0:res_x:delta[1]
    ].transpose(1, 2, 0) % 1

    # --------------------------------------------------------
    # 随机梯度
    # --------------------------------------------------------

    angles = (
        2
        * np.pi
        * np.random.rand(
            res_y + 1,
            res_x + 1
        )
    )

    gradients = np.dstack(
        (
            np.cos(angles),
            np.sin(angles)
        )
    )

    g00 = (
        gradients[:-1, :-1]
        .repeat(d[0], axis=0)
        .repeat(d[1], axis=1)
    )

    g10 = (
        gradients[1:, :-1]
        .repeat(d[0], axis=0)
        .repeat(d[1], axis=1)
    )

    g01 = (
        gradients[:-1, 1:]
        .repeat(d[0], axis=0)
        .repeat(d[1], axis=1)
    )

    g11 = (
        gradients[1:, 1:]
        .repeat(d[0], axis=0)
        .repeat(d[1], axis=1)
    )

    # --------------------------------------------------------
    # Ramp
    # --------------------------------------------------------

    n00 = np.sum(
        grid * g00,
        axis=2
    )

    n10 = np.sum(
        np.dstack(
            (
                grid[:, :, 0] - 1,
                grid[:, :, 1]
            )
        ) * g10,
        axis=2
    )

    n01 = np.sum(
        np.dstack(
            (
                grid[:, :, 0],
                grid[:, :, 1] - 1
            )
        ) * g01,
        axis=2
    )

    n11 = np.sum(
        np.dstack(
            (
                grid[:, :, 0] - 1,
                grid[:, :, 1] - 1
            )
        ) * g11,
        axis=2
    )

    # --------------------------------------------------------
    # 插值
    # --------------------------------------------------------

    t = f(grid)

    n0 = (
        n00 * (1 - t[:, :, 0])
        +
        t[:, :, 0] * n10
    )

    n1 = (
        n01 * (1 - t[:, :, 0])
        +
        t[:, :, 0] * n11
    )

    noise = np.sqrt(2) * (
        (1 - t[:, :, 1]) * n0
        +
        t[:, :, 1] * n1
    )

    # --------------------------------------------------------
    # 裁剪回原始尺寸
    # --------------------------------------------------------

    noise = noise[
        :original_h,
        :original_w
    ]

    # --------------------------------------------------------
    # 归一化
    # --------------------------------------------------------

    noise_min = noise.min()
    noise_max = noise.max()

    if noise_max > noise_min:

        noise = (
            noise - noise_min
        ) / (
            noise_max - noise_min
        )

    else:

        noise = np.zeros_like(noise)

    return noise.astype(np.float32)


# ============================================================
# Cloud Occlusion
# ============================================================

def add_cloud_occlusion(
        image_float,
        cloud_coverage=0.40,
        cloud_opacity=0.75,
        num_clouds=8,
        min_size=30,
        max_size=120,
        perlin_strength=0.30,
        perlin_resolution=(4, 4),
        dtype_max=255.0
):
    """
    添加云遮挡。

    Parameters
    ----------
    cloud_coverage :
        云覆盖程度。

        推荐：
        0.20 -> 较少
        0.30 -> 轻中度
        0.40 -> 中度
        0.55 -> 较多
        0.70 -> 大面积覆盖

    cloud_opacity :
        云的不透明程度。

        推荐：
        0.40 -> 很薄
        0.55 -> 薄云
        0.70 -> 中等
        0.85 -> 厚云
        0.95 -> 接近完全遮挡

    num_clouds :
        云团数量。

    min_size / max_size :
        云团尺寸。

        数值越大，云团越大。

    perlin_strength :
        控制云边缘不规则程度。
    """

    h, w = image_float.shape[:2]

    cloud_mask = np.zeros(
        (h, w),
        dtype=np.float32
    )

    # 只生成一次坐标矩阵，提高效率
    y, x = np.ogrid[
        :h,
        :w
    ]

    # --------------------------------------------------------
    # Step 1
    # 随机生成多个云核
    # --------------------------------------------------------

    for _ in range(num_clouds):

        cx = np.random.randint(
            0,
            w
        )

        cy = np.random.randint(
            0,
            h
        )

        radius = np.random.uniform(
            min_size,
            max_size
        )

        dist = np.sqrt(
            (x - cx) ** 2
            +
            (y - cy) ** 2
        )

        cloud_blob = np.exp(
            -dist ** 2
            /
            (
                2
                *
                (radius / 2) ** 2
            )
        )

        cloud_mask = np.maximum(
            cloud_mask,
            cloud_blob
        )

    # --------------------------------------------------------
    # Step 2
    # Perlin Noise 破坏规则圆形结构
    # --------------------------------------------------------

    perlin = generate_perlin_noise_2d(
        (h, w),
        perlin_resolution
    )

    cloud_mask = np.clip(
        cloud_mask
        +
        perlin_strength
        *
        (perlin - 0.5),
        0,
        1
    )

    # --------------------------------------------------------
    # Step 3
    # 根据目标 coverage 调整总体云量
    # --------------------------------------------------------

    current_coverage = np.mean(
        cloud_mask > 0.1
    )

    if current_coverage > 0:

        scale_factor = (
            cloud_coverage
            /
            current_coverage
        )

        cloud_mask = np.clip(
            cloud_mask * scale_factor,
            0,
            1
        )

    # --------------------------------------------------------
    # Step 4
    # 设置透明度
    # --------------------------------------------------------

    cloud_alpha = (
        cloud_mask
        *
        cloud_opacity
    )

    # --------------------------------------------------------
    # Step 5
    # 云颜色
    # --------------------------------------------------------

    cloud_color = np.full_like(
        image_float,
        dtype_max * 0.95
    )

    alpha = np.expand_dims(
        cloud_alpha,
        axis=-1
    )

    cloudy = (
        image_float * (1 - alpha)
        +
        cloud_color * alpha
    )

    return cloudy


# ============================================================
# 保存 GeoTIFF
# ============================================================

def save_tif(
        image_float,
        profile,
        output_path
):
    """
    保留原始 TIFF 的投影、坐标、transform 等信息。
    """

    image_float = np.clip(
        image_float,
        0,
        255
    )

    image_out = image_float.astype(
        profile['dtype']
    )

    # H W C -> C H W
    image_chw = np.transpose(
        image_out,
        (2, 0, 1)
    )

    with rasterio.open(
            output_path,
            'w',
            **profile
    ) as dst:

        dst.write(
            image_chw
        )


# ============================================================
# 主函数
# ============================================================

def process_single_image(
        input_path,
        output_dir,

        # ---------------------
        # Blur 参数
        # ---------------------
        blur_w=5,

        # ---------------------
        # Gaussian Noise 参数
        # ---------------------
        noise_sigma=20,

        # ---------------------
        # Stripe 参数
        # ---------------------
        stripe_r=0.20,
        stripe_probability=0.10,
        stripe_direction='horizontal',

        # ---------------------
        # Fog 参数
        # ---------------------
        fog_A=0.90,
        fog_t0=0.60,

        # ---------------------
        # Cloud 参数
        # ---------------------
        cloud_coverage=0.40,
        cloud_opacity=0.75,
        num_clouds=8,
        cloud_min_size=30,
        cloud_max_size=120,
        cloud_perlin_strength=0.30
):

    # ========================================================
    # 检查输入文件
    # ========================================================

    if not os.path.isfile(input_path):

        raise FileNotFoundError(
            f"找不到输入图像：\n{input_path}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # 获取文件名称
    #
    # 例如：
    # 166.tif
    #
    # base_name = 166
    # ========================================================

    filename = os.path.basename(
        input_path
    )

    base_name, ext = os.path.splitext(
        filename
    )

    # ========================================================
    # 读取图像
    # ========================================================

    print("=" * 60)

    print(
        f"读取图像：{input_path}"
    )

    with rasterio.open(
            input_path
    ) as src:

        profile = src.profile.copy()

        img_data = src.read()

        print(
            f"图像尺寸："
            f"{src.width} × {src.height}"
        )

        print(
            f"波段数量："
            f"{src.count}"
        )

        print(
            f"数据类型："
            f"{src.dtypes[0]}"
        )

    # C H W -> H W C
    img = np.transpose(
        img_data,
        (1, 2, 0)
    )

    # 所有退化都使用 float32
    img_float = img.astype(
        np.float32
    )

    # ========================================================
    # 1. Blur
    # ========================================================

    print(
        "\n[1/5] Generating Blur..."
    )

    blur_img = add_blur(
        img_float,
        w=blur_w
    )

    blur_path = os.path.join(
        output_dir,
        f"{base_name}_blur{ext}"
    )

    save_tif(
        blur_img,
        profile,
        blur_path
    )

    print(
        f"保存：{blur_path}"
    )

    # ========================================================
    # 2. Gaussian Noise
    # ========================================================

    print(
        "\n[2/5] Generating Gaussian Noise..."
    )

    noise_img = add_gaussian_noise(
        img_float,
        sigma=noise_sigma
    )

    noise_path = os.path.join(
        output_dir,
        f"{base_name}_noise{ext}"
    )

    save_tif(
        noise_img,
        profile,
        noise_path
    )

    print(
        f"保存：{noise_path}"
    )

    # ========================================================
    # 3. Stripe
    # ========================================================

    print(
        "\n[3/5] Generating Stripe Noise..."
    )

    stripe_img = add_stripe_noise(
        img_float,
        r=stripe_r,
        probability=stripe_probability,
        direction=stripe_direction
    )

    stripe_path = os.path.join(
        output_dir,
        f"{base_name}_stripe{ext}"
    )

    save_tif(
        stripe_img,
        profile,
        stripe_path
    )

    print(
        f"保存：{stripe_path}"
    )

    # ========================================================
    # 4. Fog
    # ========================================================

    print(
        "\n[4/5] Generating Fog..."
    )

    fog_img = add_fog(
        img_float,
        A=fog_A,
        t0=fog_t0
    )

    fog_path = os.path.join(
        output_dir,
        f"{base_name}_fog{ext}"
    )

    save_tif(
        fog_img,
        profile,
        fog_path
    )

    print(
        f"保存：{fog_path}"
    )

    # ========================================================
    # 5. Cloud
    # ========================================================

    print(
        "\n[5/5] Generating Cloud Occlusion..."
    )

    cloud_img = add_cloud_occlusion(
        img_float,
        cloud_coverage=cloud_coverage,
        cloud_opacity=cloud_opacity,
        num_clouds=num_clouds,
        min_size=cloud_min_size,
        max_size=cloud_max_size,
        perlin_strength=cloud_perlin_strength
    )

    cloud_path = os.path.join(
        output_dir,
        f"{base_name}_cloud{ext}"
    )

    save_tif(
        cloud_img,
        profile,
        cloud_path
    )

    print(
        f"保存：{cloud_path}"
    )

    # ========================================================
    # 完成
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "✅ 5 种遥感图像退化全部生成完成！"
    )

    print(
        "=" * 60
    )

    print(
        "\n生成文件："
    )

    print(
        f"1. {base_name}_blur{ext}"
    )

    print(
        f"2. {base_name}_noise{ext}"
    )

    print(
        f"3. {base_name}_stripe{ext}"
    )

    print(
        f"4. {base_name}_fog{ext}"
    )

    print(
        f"5. {base_name}_cloud{ext}"
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == '__main__':

    # ========================================================
    # 只需要重点修改下面这些参数
    # ========================================================

    # --------------------------------------------------------
    # 输入：某一张 TIFF
    # --------------------------------------------------------

    input_path = (
        r"/hdd2/BoYu/data/HL/test/img/167.tif"
    )

    # --------------------------------------------------------
    # 输出文件夹
    # --------------------------------------------------------

    output_dir = (
        r"/hdd2/BoYu/data/HL/data3/"
    )

    # ========================================================
    # Blur
    # ========================================================

    blur_w = 6

    # ========================================================
    # Gaussian Noise
    # ========================================================

    noise_sigma = 25

    # ========================================================
    # Stripe Noise
    # ========================================================

    stripe_r = 0.30

    stripe_probability = 0.20

    stripe_direction = 'horizontal'

    # ========================================================
    # Fog
    # ========================================================

    fog_A = 0.95

    fog_t0 = 0.50

    # ========================================================
    # Cloud
    # ========================================================

    cloud_coverage = 0.50

    cloud_opacity = 0.8

    num_clouds = 8

    cloud_min_size = 30

    cloud_max_size = 120

    cloud_perlin_strength = 0.30

    # ========================================================
    # 开始处理
    # ========================================================

    process_single_image(

        input_path=input_path,

        output_dir=output_dir,

        # Blur
        blur_w=blur_w,

        # Gaussian Noise
        noise_sigma=noise_sigma,

        # Stripe
        stripe_r=stripe_r,
        stripe_probability=stripe_probability,
        stripe_direction=stripe_direction,

        # Fog
        fog_A=fog_A,
        fog_t0=fog_t0,

        # Cloud
        cloud_coverage=cloud_coverage,
        cloud_opacity=cloud_opacity,
        num_clouds=num_clouds,
        cloud_min_size=cloud_min_size,
        cloud_max_size=cloud_max_size,
        cloud_perlin_strength=cloud_perlin_strength
    )