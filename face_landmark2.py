# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:13:50 2018

@author: ai
"""

import dlib
import numpy
import cv2
from skimage import io

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_path = "test1.jpg"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
img = io.imread(faces_path)

win.clear_overlay()
win.set_image(img)

dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

for i, d in enumerate(dets):
    shape = predictor(img, d) 
    landmark = numpy.matrix([[p.x, p.y] for p in shape.parts()])
    win.add_overlay(shape) #画出landmark
    
for idx, point in enumerate(landmark):
    print("point",point)
    print("type point",point.shape)
    pos = (point[0, 0], point[0, 1]) #上述生成landmark那行可以看出point维度是(1,2)的，如：point [[270 221]]
    print(point[0,0])
    cv2.putText(img,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.3,color=(0,255,0))
    #fontFace 是字体，fontScale是字体大小，pos是左下坐标
    cv2.circle(img, pos, 1, color=(0,0,255) )

win.set_image(img)
win.add_overlay(dets)
dlib.hit_enter_to_continue()


# landmark也可以用numpy.array()函数生成，建议用numpy
# landmark = numpy.array([[p.x, p.y] for p in shape.parts()])
# 相对应的，pos = (point[0], point[1])
# 格式如：point [232,221]
