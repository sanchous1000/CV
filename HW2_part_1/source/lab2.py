# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:41:29 2024

@author: Ivan
"""

import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

def template_search(input_image, template, metric):
    img_full=cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
    img_cut=cv.cvtColor(template, cv.COLOR_RGB2GRAY)
    matching_output = cv.matchTemplate(img_full, img_cut, metric)
    least_value, peak_value, least_coord, peak_coord = cv.minMaxLoc(matching_output)
    highlight_start = peak_coord
    pattern_w, pattern_h = img_cut.shape[::-1]
    highlight_end = (highlight_start[0] + pattern_w, highlight_start[1] + pattern_h)
    output=input_image.copy()
    cv.rectangle(output, highlight_start, highlight_end, 255, 2)
    return output

img_full=cv.imread('mbappe.jpg')
img_cut=cv.imread('mbappe_cut.jpg')

methods_names = ['cv.TM_CCORR','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
methods = [cv.TM_CCORR,cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
#Определение метрики
for method in range(len(methods)):
    output_image=template_search(img_full,img_cut,methods[method])
    plt.imshow(output_image[:,:,::-1])
    plt.title(methods_names[method])
    plt.show()
    

# Зашумлённая картинка
noise = np.zeros(img_full.shape, np.uint8)
cv.randn(noise, 0, 700)
noisy_img = cv.add(img_full, noise)
output_image=template_search(noisy_img,img_cut,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('TM_CCORR_NORMED')
plt.show()

#Перевёрнутая картинка
img_reversed=cv.imread('mbappe_reversed.jpg')
output_image=template_search(img_reversed,img_cut,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('TM_CCORR_NORMED')
plt.show()

#Отзеркаленная картинка
img_mirrored=cv.imread('mbappe_mirrored.jpg')
output_image=template_search(img_mirrored,img_cut,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('TM_CCORR_NORMED')
plt.show()

#Кошка
img_1=cv.imread('cat_2.jpg')
img_2=cv.imread('cat_cut.jpg')
output_image=template_search(img_1,img_2,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('TM_CCORR_NORMED')
plt.show()
plt.imshow(img_2[:,:,::-1])
plt.show()

#Цветок
img_1=cv.imread('flower_2.jpg')
img_2=cv.imread('flower_cut.jpg')
output_image=template_search(img_1,img_2,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('TM_CCORR_NORMED')
plt.show()
plt.imshow(img_2[:,:,::-1])
plt.show()

#Лиса
img_1=cv.imread('fox.jpg')
img_2=cv.imread('fox_cut.jpg')
output_image=template_search(img_1,img_2,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('cv.TM_CCORR_NORMED')
plt.show()
plt.imshow(img_2[:,:,::-1])
plt.show()

#Домик
img_1=cv.imread('house_2.jpg')
img_2=cv.imread('house_cut.jpg')
output_image=template_search(img_1,img_2,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('cv.TM_CCORR_NORMED')
plt.show()
plt.imshow(img_2[:,:,::-1])
plt.show()

#Картина
img_1=cv.imread('photo_2.jpg')
img_2=cv.imread('photo_cut.jpg')
output_image=template_search(img_1,img_2,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title(methods_names[method])
plt.show()
plt.imshow(img_2[:,:,::-1])
plt.show()

#Маска
img_1=cv.imread('mask_2.jpg')
img_2=cv.imread('mask_cut.jpg')
output_image=template_search(img_1,img_2,cv.TM_CCORR_NORMED)
plt.imshow(output_image[:,:,::-1])
plt.title('cv.TM_CCORR_NORMED')
plt.show()
plt.imshow(img_2[:,:,::-1])
plt.show()