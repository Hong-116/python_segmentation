import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from skimage import morphology
import numpy as np

from skimage import color
def prune_skeleton(image, iterations=5):
    """
    去除图像骨架中的毛刺。

    参数:
    - image: 输入的二值图像骨架。
    - iterations: 修剪迭代次数。

    返回:
    - 修剪后的图像。
    """
    # 复制图像以进行处理
    pruned_img = image.copy()

    # 定义一个3x3的十字形结构元素
    cross_kernel = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)

    for _ in range(iterations):
        # 查找所有端点：骨架像素，其仅有一个8邻域像素
        endpoints = cv2.filter2D(pruned_img, -1, cross_kernel) == 2

        # 去除端点
        pruned_img[endpoints] = 0

        # 如果没有端点被去除，则终止迭代
        if not np.any(endpoints):
            break

    return pruned_img

def prune_skeleton2(skeleton, pruning_size=3):
    """
    This function prunes the skeleton by removing spurs of the specified size.
    :param skeleton: binary image of the skeleton
    :param pruning_size: size of the pruning element
    :return: pruned skeleton
    """
    # 创建用于修剪的十字形结构元素
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (pruning_size, pruning_size))

    # 迭代应用修剪大小
    pruned_skeleton2 = skeleton.copy()
    while True:
        # 在修剪之前存储骨架的副本以供比较
        prev_skeleton2 = pruned_skeleton2.copy()

        # 侵蚀骨架
        eroded = cv2.erode(pruned_skeleton2, element)

        # 扩张被侵蚀的骨架以恢复主体结构
        temp = cv2.dilate(eroded, element)

        # 从原图中减去膨胀得到终点
        end_points = cv2.subtract(pruned_skeleton2, temp)

        # 从骨架中减去端点以移除它们
        pruned_skeleton2 = cv2.subtract(pruned_skeleton2, end_points)

        # 如果没有变化，则修剪完成
        if cv2.countNonZero(pruned_skeleton2 - prev_skeleton2) == 0:
            break

    return pruned_skeleton2

# 去除孤立点
def Img1(src):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8, ltype=None)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  # 创建个全0的黑背景
    for i in range(1, num_labels):
        mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] > 10:  # 300是面积 可以随便调
            img[mask] = 255
            # 面积大于300的区域涂白留下，小于300的涂0抹去
        else:
            img[mask] = 0

    return img

# 端点检测
def detect_endpoints(skel):
    # 定义检测端点的内核
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # 过滤骨架图像
    filtered = cv2.filter2D(skel, -1, kernel)

    # 查找端点：只有一个邻居的像素
    endpoints = np.where(filtered == 11)
    return zip(endpoints[1], endpoints[0])  # x, y coordinates

# # 批量预测
# # 设置图片的源目录和目标目录
# source_dir = 'D:/python_object/mask_rcnn/mask_result/mask/'
# binary_dir = 'D:/python_object/mask_rcnn/mask_result/binary/'
# skeleton_dir = 'D:/python_object/mask_rcnn/mask_result/skeleton/'
# mask_dir = 'D:/python_object/mask_rcnn/mask_result/denoising/'

# # 创建目标目录，如果不存在的话
# os.makedirs(binary_dir, exist_ok=True)
# os.makedirs(skeleton_dir, exist_ok=True)

# # 获取所有JPG文件的列表
# image_files = glob(os.path.join(source_dir, '*.JPG'))

# for image_file in image_files:
#     img = mpimg.imread(image_file)

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     low_hsv = np.array([35, 43, 46])
#     high_hsv = np.array([77, 255, 255])
#     mask0 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

#     # 去除高斯噪声
#     # mask = cv2.GaussianBlur(mask0, (5, 5), 0)
#     # 中值滤波
#     mask = cv2.medianBlur(mask0, 13)

#     base_name = os.path.basename(image_file)
#     mask_image_path = os.path.join(mask_dir, base_name.replace('.JPG', '.binary_image.png'))
#     cv2.imwrite(mask_image_path, mask)

#     # 生成二值图像
#     binary_image = np.where(mask, 255, 0).astype(np.uint8)

#     # 生成输出路径

#     binary_image_path = os.path.join(binary_dir, base_name.replace('.JPG', '.binary_image.png'))
#     cv2.imwrite(binary_image_path, binary_image)

#     binary_image[binary_image == 255] = 1

#     skeleton0 = morphology.skeletonize(binary_image)  # 骨架提取
#     skeleton = skeleton0.astype(np.uint8) * 255
#     skeleton_path = os.path.join(skeleton_dir, base_name.replace('.JPG', '.skeleton.png'))
#     cv2.imwrite(skeleton_path, skeleton)  # 保存骨架提取后的图片

# print("处理完成！所有图片已经保存到指定文件夹。")


# 单张预测
img = mpimg.imread('D:/python_object/mask_rcnn/mask_result/mask/test.JPG')  # 读取图片

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
low_hsv = np.array([35, 43, 46])
high_hsv = np.array([77, 255, 255])
mask0 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

# 去除高斯噪声
# mask = cv2.GaussianBlur(mask0, (5, 5), 0)

# 中值滤波
mask = cv2.medianBlur(mask0, 13)

# Convert the mask to a binary image
binary_image = np.where(mask, 255, 0).astype(np.uint8)
binary_image_path = "D:/python_object/mask_rcnn/mask_result/binary/test_test.binary_image.png"
cv2.imwrite(binary_image_path, binary_image)

binary_image[binary_image == 255] = 1

skeleton0 = morphology.skeletonize(binary_image)  # 骨架提取
skeleton = skeleton0.astype(np.uint8) * 255
skeleton_path = "D:/python_object/mask_rcnn/mask_result/skeleton/test_test.skeleton.png"
cv2.imwrite(skeleton_path, skeleton)  # 保存骨架提取后的图片



# # 检测端点
# endpoints = detect_endpoints(skeleton)

# # 创建彩色输出图片，标记端点
# colored_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

# # 用红色标记镜像上的端点
# for x, y in endpoints:
#     cv2.circle(colored_skeleton, (x, y), 5, (0, 0, 255), -1)

# # 保存图片
# cv2.imwrite('D:/python_object/mask_rcnn/mask_result/skeleton_endpoints/test_endpoints.png', colored_skeleton)

# # 打印端点坐标
# for point in endpoints:
#     print(point)


