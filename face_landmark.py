# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:13:50 2018

@author: ai
"""

import dlib
import numpy
from skimage import io

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_path = "test1.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
print(predictor)
type(predictor)


win = dlib.image_window()
img = io.imread(faces_path)

#Remove all overlays from the image_window.
win.clear_overlay()

#Make the image_window display the given image.
win.set_image(img)

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

for i, d in enumerate(dets):
    print(d)
    shape = predictor(img, d)
    print("shape:", shape)
    print("shape.parts()",shape.parts())
    # shape = predictor(img, dets[i])
    landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    print("landmark",landmark) #模型有68个标记点
    win.add_overlay(shape)

#绘制矩阵轮廓
win.add_overlay(dets)

#保持图像
dlib.hit_enter_to_continue()
