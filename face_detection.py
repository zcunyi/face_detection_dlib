# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:48:40 2018

@author: ai
"""

import dlib
from skimage import io

#使用Dlib的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()

#生成Dlib的图像窗口
window = dlib.image_window()
img = io.imread("test2.jpg")
# io.imshow(img)

dets = detector(img, 1) #dets 返回结果是人脸数和相关坐标
print("Number of faces detected: {}".format(len(dets))) 
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))

#Remove all overlays from the image_window.
window.clear_overlay()

#Make the image_window display the given image.
window.set_image(img)

#绘制矩阵轮廓
window.add_overlay(dets)

#保持图像
dlib.hit_enter_to_continue()

