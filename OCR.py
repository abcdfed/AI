#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from PIL import Image
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取文件
# imagePath = "C:\\Users\\DELL\\Desktop\\P/666.jpg"
imagePath = "C:\\Users\\DELL\\Desktop\\P/156.png"
img = cv2.imread(imagePath)

# 灰度显示
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 此步骤形态学变换的预处理，得到可以查找矩形的图片
# 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
# 二值化
ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

# 设置膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

# 膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations = 1)

# 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
erosion = cv2.erode(dilation, element1, iterations = 1)

# 再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2, iterations = 3)

# # 显示连续膨胀3次后的效果
# plt.imshow(dilation2,'gray')
# plt.show()

#  查找和筛选文字区域
region = []
#  查找轮廓
contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 利用以上函数可以得到多个轮廓区域，存在一个列表中。
#  筛选那些面积小的
for i in range(len(contours)):
    # 遍历所有轮廓
    # cnt是一个点集
    cnt = contours[i]

    # 计算该轮廓的面积
    area = cv2.contourArea(cnt)

    # 面积小的都筛选掉、这个1000可以按照效果自行设置
    if(area < 1000):
        continue

    # 找到最小的矩形，该矩形可能有方向
    rect = cv2.minAreaRect(cnt)

    # box是四个点的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)
	# 仅输出汉字框  顺序从下往上 
    print("rect is:")
    print(box)

    # 计算高和宽
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])

    # 筛选那些太细的矩形，留下扁的
    if(height > width * 1.3):
        continue

    region.append(box)

# 用绿线画出这些找到的轮廓
for box in region:
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

plt.imshow(img,'brg')
plt.show()
