## ORB

### 1. **Обнаружение ключевых точек на обоих изображениях**
Код для обнаружения ключевых точек и их дескрипторов:
```python
queryKeypoints, queryDescriptors = orb.detectAndCompute(main_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(example_bw, None)
```

---

### 2. **FAST: Обнаружение ключевых точек**
Алгоритм FAST используется для определения ключевых точек на изображении. Ключевая точка определяется по изменению интенсивностей на окружности радиуса \( r \) вокруг точки \( I_p \).

#### **Условие ключевой точки:**
![image](https://github.com/user-attachments/assets/8a66aa73-2503-4b19-b111-070bc6baa8df)

Где:
- \( I_p \) — интенсивность центрального пикселя,
- \( I_i \) — интенсивности пикселей на окружности,
- \( t \) — пороговое значение.

---
### 3. **BRIEF: Бинарные дескрипторы и фильтрация ключевых точек**
BRIEF — это алгоритм, используемый для описания ключевых точек, создавая бинарные дескрипторы.

#### **Мера Харриса:**
Формируется матрица автокорреляции A (матрица второго момента)

![alt text](image-1.png)
![alt text](image-2.png)

Мера Харриса: 
![alt text](image-3.png), 

альфа - константа

Результатом является бинарный дескриптор, объединяющий результаты таких сравнений.

---

### 4. **Сопоставление ключевых точек**
Сопоставление дескрипторов выполняется с использованием расстояния Хэмминга, которое подсчитывает количество отличий между бинарными строками.

Код:
```python
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
```

Сопоставленные точки сортируются по расстоянию, чтобы выделить лучшие совпадения.

---

### 5. **Гомография**
Гомография связывает два изображения, учитывая их пространственное положение. Это преобразование описывается матрицей \( H \) размером 3x3, которая переводит точки \( (x_1, y_1) \) на первом изображении в \( (x_2, y_2) \) на втором:

![alt text](image.png)
Где:
- \( H \) — матрица гомографии.

---

### 6. **Изменение перспективы**
С помощью матрицы гомографии выполняется преобразование перспективы, чтобы одно изображение совпало с другим:

p2 = H * p1, где 
p1 - координаты точки на исходном изображении, p2 - точки на новом изображении.
---

### **Визуализация совпадений и результатов**
Матрица гомографии используется для изменения перспективы и выделения области на изображении. 

**Пример визуализации:**

![Результаты](sample1.jpg)
![Результаты](sample2.jpg)

---

### Используемые источники

| №  | Источник                                                                                 | Описание                                   |
|----|------------------------------------------------------------------------------------------|-------------------------------------------|
| 1  | [Формулы и объяснение ORB](https://medium.com/@imhongxiaohui/explanation-of-orb-in-point-feature-extraction-1cdd9b82655a)       | Принцип работы алгоритма |
| 2  | [Статья про ORB](https://habr.com/ru/articles/414459/) | Принцип работы алгоритма    |
| 3  | [Гомография](https://waksoft.susu.ru/2020/03/26/primery-gomogrfii-s-ispolzovaniem-opencv/) | Гомография |
