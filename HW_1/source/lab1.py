# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 01:21:44 2024

@author: Ivan
"""

import cv2 as cv
import numpy as np  
import matplotlib.pyplot as plt 
import time

def im2col(X, kernel_size):#Функция, позволяющая развернуть элементы, по которым идёт проход ядром, в столбцы матрицы
    result = np.empty([X.shape[0],kernel_size[0] * kernel_size[1], (X.shape[1] - kernel_size[0] + 1)*(X.shape[2] - kernel_size[1] + 1)])
    for i in range(X.shape[2] - kernel_size[1]+1):
        for j in range(X.shape[1] - kernel_size[0]+1):
            result[:,:, i *(X.shape[1] - kernel_size[0] + 1)+ j] = X[:,j:j + kernel_size[0], i:i + kernel_size[1]].reshape(X.shape[0],kernel_size[0] * kernel_size[1])
    return result

class GaussianFilter:
    def _init_weights_matrix(self, size=(3,3), std=1):
        im_vert=np.repeat(np.arange(size[0]),size[1]).reshape(size[0],size[1])
        im_vert=np.exp(-np.square(im_vert-((size[0]-1)//2))/(2*std**2))#Получение весов по вертикали
        im_hor=np.repeat(np.arange(size[1]),size[0]).reshape(size[1],size[0]).T
        im_hor=np.exp(-np.square(im_hor-((size[1]-1)//2))/(2*std**2))#Получение весов по горизонтали
        result=(im_vert*im_hor)/(2*np.pi*std**2)#Перемножение весов
        return result/np.sum(result)#Возвращение нормализованныз весов
    def transform(self, image, size=(3,3), std=1):
        W=self._init_weights_matrix(size, std)
        img=cv.imread(image)
        img=img.transpose(2,0,1)#Транспонирование изображения для умножения на весы
        col_img=im2col(img,size)#Приведение значений пикселей в столбик
        img_itog=np.matmul(W.reshape(1,1,size[0]*size[1]),col_img).astype(int)#Получение столбца с преобразованными значениями
        return img_itog.reshape(img.shape[0],img.shape[1]-size[0]+1,img.shape[2]-size[1]+1, order='F').transpose(1,2,0) #Возвращение преобразованной картинки
    
gf=GaussianFilter()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axes[0, 0].imshow(cv.imread('dog.jpg')[:, :, ::-1])
start = time.time()
axes[0, 1].imshow(gf.transform('dog.jpg', size=(25,25),std=3)[:, :, ::-1])
end = time.time()
axes[1, 0].imshow(gf.transform('dog.jpg', size=(25,25),std=5)[:, :, ::-1])
axes[1, 1].imshow(gf.transform('dog.jpg', size=(25,25),std=10)[:, :, ::-1])
axes[0, 0].set_title('Оригинальное фото')
axes[0, 1].set_title('Размытое фото, ядро 25x25, std=3')
axes[1,0].set_title('Размытое фото, ядро 25x25, std=5')
axes[1, 1].set_title('Размытое фото, ядро 25x25, std=10')
plt.show()
#plt.savefig('dog_result.png')

print("Время выполнения написанного алгоритма (для картинки высокого качества):", (end-start) * 10**3, "ms")

image = cv.imread('dog.jpg')
start = time.time()
img_blur_3 = cv.GaussianBlur(image, (25,25), 10)
end = time.time()
print("Время выполнения эталонного алгоритма (для картинки высокого качества):", (end-start) * 10**3, "ms")
plt.imshow(img_blur_3[:,:,::-1])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axes[0, 0].imshow(cv.imread('smile.jpg')[:, :, ::-1])
start = time.time()
axes[0, 1].imshow(gf.transform('smile.jpg', size=(11,11),std=3)[:, :, ::-1])
end = time.time()
axes[1, 0].imshow(gf.transform('smile.jpg', size=(11,11),std=5)[:, :, ::-1])
axes[1, 1].imshow(gf.transform('smile.jpg', size=(11,11),std=10)[:, :, ::-1])
axes[0, 0].set_title('Оригинальное фото')
axes[0, 1].set_title('Размытое фото, ядро 11x11, std=3')
axes[1,0].set_title('Размытое фото, ядро 11x11, std=5')
axes[1, 1].set_title('Размытое фото, ядро 11x11, std=10')
#plt.savefig('smile_result.png')
plt.show()

print("Время выполнения написанного алгоритма (для картинки низкого качества):", (end-start) * 10**3, "ms")

image = cv.imread('smile.jpg')
start = time.time()
img_blur_3 = cv.GaussianBlur(image, (11,11), 10)
plt.imshow(img_blur_3[:,:,::-1])
end = time.time()
#plt.savefig('smile_etalon.png')
plt.show()
print("Время выполнения эталонного алгоритма (для картинки низкого качества):", (end-start) * 10**3, "ms")