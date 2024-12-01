## ORB
### 1. **Ключевые точки на обоих изображениях**
В коде:
```python
queryKeypoints, queryDescriptors = orb.detectAndCompute(main_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(example_bw, None)
```

### 2.  **FAST**:
Ключевая точка определяется на основе изменения интенсивностей на окружности радиуса \( r \) вокруг точки \( I_p \).

**Условие ключевой точки:**
![image](https://github.com/user-attachments/assets/8a66aa73-2503-4b19-b111-070bc6baa8df)

где:
- \( I_p \) — интенсивность центрального пикселя,
- \( I_i \) — интенсивности пикселей на окружности,
- \( t \) — пороговое значение.



### 3. **BRIEF для бинарных дескрипторов**
**BRIEF — бинаризация патча вокруг ключевой точки:**
Для двух точек \( x \) и \( y \) в патче:
![image](https://github.com/user-attachments/assets/75cfa0c6-62b2-4975-9a58-a3006f6e9a30)

Здесь \( p(x) \) и \( p(y) \) — интенсивности пикселей \( x \) и \( y \).


### 4. **Сопоставление ключевых точек используется рассстояние **Хамминга****
В коде:
```python
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
```

 для сравнения бинарных дескрипторов:

### 5. **Визуализация совпадений**
В коде:
```python
final_img = cv2.drawMatches(main, queryKeypoints, example, trainKeypoints, matches[:20], None)
```



**Визуализация результатов:**
Результаты выводятся в сетке \( \text{rows} \times \text{cols} \), где каждая ячейка — это результат сопоставления двух изображений.

### Пример визуализации:
![Пример визуализации](Figure_1.png)



## Используемые источники

| №  | Источник                                                                                 | Описание                                   |
|----|------------------------------------------------------------------------------------------|-------------------------------------------|
| 1  | [Формулы и объяснение ORB](https://medium.com/@imhongxiaohui/explanation-of-orb-in-point-feature-extraction-1cdd9b82655a)       | Принцип работы алгоритма |
| 2  | [Статья про ORB](https://habr.com/ru/articles/414459/) | Принцип работы алгоритма    |


