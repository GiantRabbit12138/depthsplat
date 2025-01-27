import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

class DepthImageProcessor:
    def __init__(self, threshold_ratio=0.8, bins=256):
        """
        初始化深度图像处理器
        :param threshold_ratio: 阈值比例（例如0.8表示80%）
        :param bins: 直方图的bin数量
        """
        self.threshold_ratio = threshold_ratio
        self.bins = bins

    def load_depth_image(self, image):
        """
        从PIL.Image加载16位深度图像
        :param image: PIL.Image对象
        :return: 16位深度图像 (numpy数组)
        """
        depth_image = np.array(image)
        if depth_image.dtype != np.uint16:
            raise ValueError("输入图像必须是16位深度图像 (uint16)")
        return depth_image

    def analyze_histogram(self, depth_image, save_path=None):
        """
        分析深度图像的直方图
        :param depth_image: 16位深度图像
        :return: 直方图值, bin边界
        """
        depth_image = np.array(depth_image)
        hist, bin_edges = np.histogram(
            depth_image.flatten(), bins=self.bins, range=(0, 65535))
        
                # 可视化直方图
        plt.figure()
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black')
        plt.title('Depth Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # 保存直方图图片
        if save_path:
            plt.savefig(save_path)
        
        return hist, bin_edges

    def find_discard_ranges(self, hist, bin_edges):
        """
        自动确定需要丢弃的区间
        :param hist: 直方图值
        :param bin_edges: bin边界
        :return: 需要丢弃的区间列表
        """
        discard_ranges = []
        total_pixels = np.sum(hist)
        threshold = self.threshold_ratio * total_pixels / len(hist)  # 每个区间的阈值

        for i in range(len(hist)):
            if hist[i] > threshold:
                range_start = bin_edges[i]
                range_end = bin_edges[i + 1]
                discard_ranges.append((range_start, range_end))

        return discard_ranges

    def discard_depth_ranges(self, depth_image, discard_ranges):
        """
        丢弃指定范围内的深度值
        :param depth_image: 16位深度图像
        :param discard_ranges: 需要丢弃的深度值范围列表
        :return: 处理后的深度图像
        """
        mask = np.zeros_like(depth_image, dtype=bool)
        for range_start, range_end in discard_ranges:
            mask |= (depth_image >= range_start) & (depth_image <= range_end)
        depth_image[mask] = 0  # 将无效值设置为0
        return depth_image

    def visualize_depth_image(self, depth_image, title="Depth Image"):
        """
        可视化深度图像
        :param depth_image: 深度图像
        :param title: 图像标题
        """
        plt.imshow(depth_image, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.show()

    def process_image(self, image, visualize=False):
        """
        处理单张深度图像
        :param image: PIL.Image对象
        :param visualize: 是否可视化处理前后的图像
        :return: 处理后的PIL.Image对象
        """
        # 加载深度图像
        depth_image = self.load_depth_image(image)

        # 分析直方图
        hist, bin_edges = self.analyze_histogram(depth_image)

        # 自动确定需要丢弃的区间
        discard_ranges = self.find_discard_ranges(hist, bin_edges)

        # 丢弃指定范围内的深度值
        processed_image = self.discard_depth_ranges(depth_image, discard_ranges)

        # 可视化处理前后的图像
        if visualize:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            self.visualize_depth_image(depth_image, title="Original Depth Image")
            plt.subplot(1, 2, 2)
            self.visualize_depth_image(processed_image, title="Processed Depth Image")
            plt.show()

        # 将处理后的图像转换为PIL.Image对象
        processed_image_pil = Image.fromarray(processed_image)
        return processed_image_pil