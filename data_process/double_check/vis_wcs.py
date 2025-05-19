import os
import matplotlib.pyplot as plt
from shapely.wkt import loads
import astropy.units as u
import numpy as np
def extract_polygons(file_path):
    """从dataset.txt文件中提取POLYGON坐标"""
    polygons = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) == 2:
                polygon_str = parts[1].strip()
                try:
                    polygon = loads(polygon_str)
                    polygons.append(polygon)
                except Exception as e:
                    print(f"解析POLYGON时出错: {e}")
    return polygons

def plot_celestial_map(polygons, output_dir):
    """使用Aitoff投影绘制天球图，并优化区域可视化"""
    # 增大图形尺寸和分辨率
    fig = plt.figure(figsize=(14, 7), dpi=300)  # 增大图形尺寸
    ax = fig.add_subplot(111, projection='aitoff')
    ax.set_title("World Coordinate System distribution", fontsize=14, pad=20)

    # 设置背景和网格线
    ax.set_facecolor('#f0f0f0')  # 浅灰色背景
    ax.grid(True, color='gray', linestyle='--', alpha=0.5)

    # 绘制多边形
    for polygon in polygons:
        if polygon.geom_type == 'Polygon':
            ra, dec = polygon.exterior.xy
            # 将RA从0°到360°转换为-180°到+180°以适配Aitoff投影
            ra_adjusted = np.array([(ra_val if ra_val <= 180 else ra_val - 360) for ra_val in ra])
            # 转换为弧度
            ra_rad = np.array([coord * u.deg.to(u.rad) for coord in ra_adjusted])
            dec_rad = np.array([coord * u.deg.to(u.rad) for coord in dec])
            
            # 放大多边形区域（可选：通过缩放坐标）
            scale_factor = 1.5  # 调整此值以控制放大倍数
            ra_center = np.mean(ra_rad)
            dec_center = np.mean(dec_rad)
            ra_scaled = ra_center + (ra_rad - ra_center) * scale_factor
            dec_scaled = dec_center + (dec_rad - dec_center) * scale_factor
            
            # 绘制多边形边线和填充区域
            ax.plot(ra_scaled, dec_scaled, color='blue', linewidth=2.5, alpha=0.9)  # 增加边线粗细
            ax.fill(ra_scaled, dec_scaled, color='blue', alpha=0.5)  # 增加填充透明度

    # 设置自定义刻度标签，将-180°到+180°映射到0°到360°
    ax.set_xticks(np.radians([-180, -120, -60, 0, 60, 120, 180]))
    ax.set_xticklabels(['0°', '60°', '120°', '180°', '240°', '300°', '360°'], fontsize=12, color='black')
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(['-60°', '-30°', '0°', '+30°', '+60°'], fontsize=12, color='black')

    # 添加轴标签
    ax.set_xlabel("RA", fontsize=12, labelpad=10)
    ax.set_ylabel("Dec", fontsize=12, labelpad=10)

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "celestial_map.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"天球图已保存至 {output_path}")

def main(dataset_path, output_dir):
    polygons = extract_polygons(dataset_path)
    if polygons:
        plot_celestial_map(polygons, output_dir)
    else:
        print("未找到有效的多边形数据。")
if __name__ == "__main__":
    dataset_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/split_file/datasetlist.txt"  # 请更新为您的dataset.txt文件路径
    output_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/wcs.png"  # 请更新为保存可视化结果的目录
    main(dataset_path, output_dir)