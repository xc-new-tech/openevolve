"""
条形码专用几何校正算法模块
专门针对条形码的几何特征进行优化的算法
"""

import numpy as np
import cv2
import math
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BarcodeGeometryCorrection:
    """条形码几何校正算法集合"""
    
    @staticmethod
    def detect_barcode_orientation(image: np.ndarray) -> Tuple[float, str]:
        """
        自动检测条形码方向
        
        Args:
            image: 输入图像
            
        Returns:
            (角度, 方向描述)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough直线检测
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            return 0.0, "unknown"
            
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # 将角度标准化到[-90, 90]范围
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
                
            angles.append(angle)
        
        # 计算主要角度
        if angles:
            # 使用直方图找到最常见的角度
            hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
            dominant_angle_idx = np.argmax(hist)
            dominant_angle = (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2
            
            # 确定方向
            if abs(dominant_angle) < 10:
                orientation = "horizontal"
            elif abs(dominant_angle - 90) < 10 or abs(dominant_angle + 90) < 10:
                orientation = "vertical"
            else:
                orientation = f"rotated_{dominant_angle:.1f}°"
                
            return dominant_angle, orientation
        
        return 0.0, "unknown"
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像校正方向
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            
        Returns:
            校正后的图像
        """
        if abs(angle) < 1:  # 角度太小，不需要旋转
            return image
            
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新的边界尺寸
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # 调整旋转中心
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def detect_barcode_region(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测条形码区域
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的条形码区域列表 [(x, y, w, h), ...]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 计算图像梯度
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        
        # 使用形态学操作连接条形码条纹
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
        
        # 腐蚀和膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(closed, kernel, iterations=4)
        dilated = cv2.dilate(eroded, kernel, iterations=4)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选条形码候选区域
        barcode_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 条形码区域的基本约束
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            rect_area = w * h
            
            # 基于面积、长宽比筛选
            if (1.5 < aspect_ratio < 15 and  # 条形码通常是长方形
                area > 1000 and  # 最小面积
                area / rect_area > 0.3):  # 填充率
                
                barcode_regions.append((x, y, w, h))
        
        return barcode_regions
    
    @staticmethod
    def perspective_correction(image: np.ndarray, 
                             corners: np.ndarray = None) -> np.ndarray:
        """
        透视变换校正
        
        Args:
            image: 输入图像
            corners: 四个角点坐标，如果为None则自动检测
            
        Returns:
            校正后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if corners is None:
            # 自动检测角点
            corners = BarcodeGeometryCorrection._detect_corners(gray)
            
        if corners is None or len(corners) != 4:
            return image
            
        # 对角点排序：左上、右上、右下、左下
        corners = BarcodeGeometryCorrection._order_points(corners)
        
        # 计算目标尺寸
        (tl, tr, br, bl) = corners
        
        # 计算宽度和高度
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # 目标点
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
        
        # 应用透视变换
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    @staticmethod
    def _detect_corners(image: np.ndarray) -> Optional[np.ndarray]:
        """检测四个角点"""
        # 边缘检测
        edges = cv2.Canny(image, 50, 150)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # 检查前5个最大轮廓
            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果是四边形
            if len(approx) == 4:
                return approx.reshape((4, 2))
        
        return None
    
    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """按左上、右上、右下、左下顺序排列点"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # 左上角有最小的和，右下角有最大的和
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 右上角有最小的差，左下角有最大的差
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    @staticmethod
    def multi_scale_detection(image: np.ndarray, 
                             scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> List[dict]:
        """
        多尺度条形码检测
        
        Args:
            image: 输入图像
            scales: 检测尺度列表
            
        Returns:
            检测结果列表，每个结果包含{scale, regions, confidence}
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        results = []
        
        for scale in scales:
            # 缩放图像
            if scale != 1.0:
                h, w = gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(gray, (new_w, new_h))
            else:
                scaled = gray.copy()
            
            # 检测条形码区域
            regions = BarcodeGeometryCorrection.detect_barcode_region(scaled)
            
            # 将坐标转换回原始尺度
            if scale != 1.0:
                original_regions = []
                for (x, y, w, h) in regions:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    original_regions.append((orig_x, orig_y, orig_w, orig_h))
                regions = original_regions
            
            # 计算置信度（基于区域数量和质量）
            confidence = len(regions) * 0.5 + min(1.0, len(regions) * 0.25)
            
            results.append({
                'scale': scale,
                'regions': regions,
                'confidence': confidence
            })
        
        return results
    
    @staticmethod
    def enhance_barcode_contrast(image: np.ndarray) -> np.ndarray:
        """
        增强条形码对比度
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 局部自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 锐化滤波
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 混合原图和锐化结果
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result


def process_barcode_geometry(image: np.ndarray,
                           operation: str = 'auto_correct',
                           **kwargs) -> np.ndarray:
    """
    条形码几何校正统一接口
    
    Args:
        image: 输入图像
        operation: 操作类型
        **kwargs: 额外参数
        
    Returns:
        处理后的图像
    """
    if operation == 'auto_correct':
        # 自动检测和校正
        angle, orientation = BarcodeGeometryCorrection.detect_barcode_orientation(image)
        
        if abs(angle) > 2:  # 只有角度足够大才进行校正
            corrected = BarcodeGeometryCorrection.rotate_image(image, -angle)
        else:
            corrected = image.copy()
            
        # 增强对比度
        enhanced = BarcodeGeometryCorrection.enhance_barcode_contrast(corrected)
        
        return enhanced
        
    elif operation == 'detect_orientation':
        angle, orientation = BarcodeGeometryCorrection.detect_barcode_orientation(image)
        logger.info(f"检测到条形码方向: {orientation}, 角度: {angle:.2f}°")
        return image
        
    elif operation == 'rotate':
        angle = kwargs.get('angle', 0)
        return BarcodeGeometryCorrection.rotate_image(image, angle)
        
    elif operation == 'detect_regions':
        regions = BarcodeGeometryCorrection.detect_barcode_region(image)
        logger.info(f"检测到 {len(regions)} 个条形码区域")
        return image
        
    elif operation == 'perspective_correct':
        corners = kwargs.get('corners', None)
        return BarcodeGeometryCorrection.perspective_correction(image, corners)
        
    elif operation == 'multi_scale':
        scales = kwargs.get('scales', [0.5, 1.0, 1.5, 2.0])
        results = BarcodeGeometryCorrection.multi_scale_detection(image, scales)
        logger.info(f"多尺度检测完成，共 {len(results)} 个尺度")
        return image
        
    elif operation == 'enhance_contrast':
        return BarcodeGeometryCorrection.enhance_barcode_contrast(image)
        
    else:
        logger.warning(f"未知的操作类型: {operation}")
        return image


def test_barcode_geometry_correction():
    """测试条形码几何校正算法"""
    import os
    
    # 创建测试图像
    test_image = np.ones((200, 400), dtype=np.uint8) * 128
    
    # 添加条形码模式
    for i in range(0, 400, 20):
        if (i // 20) % 2 == 0:
            test_image[:, i:i+10] = 0
        else:
            test_image[:, i:i+10] = 255
    
    print("测试条形码几何校正算法...")
    
    # 测试方向检测
    angle, orientation = BarcodeGeometryCorrection.detect_barcode_orientation(test_image)
    print(f"方向检测: {orientation}, 角度: {angle:.2f}°")
    
    # 测试旋转校正
    rotated = BarcodeGeometryCorrection.rotate_image(test_image, 15)
    print(f"旋转校正: {rotated.shape}")
    
    # 测试区域检测
    regions = BarcodeGeometryCorrection.detect_barcode_region(test_image)
    print(f"区域检测: 找到 {len(regions)} 个区域")
    
    # 测试多尺度检测
    results = BarcodeGeometryCorrection.multi_scale_detection(test_image)
    print(f"多尺度检测: {len(results)} 个尺度结果")
    
    # 测试统一接口
    processed = process_barcode_geometry(test_image, 'auto_correct')
    print(f"统一接口: {processed.shape}")
    
    print("所有条形码几何校正算法测试完成！")


if __name__ == "__main__":
    test_barcode_geometry_correction() 