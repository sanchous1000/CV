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
\[
I_i > I_p + t \quad \text{или} \quad I_i < I_p - t,
\]
где:
- \( I_p \) — интенсивность центрального пикселя,
- \( I_i \) — интенсивности пикселей на окружности,
- \( t \) — пороговое значение.

---

### 3. **BRIEF для бинарных дескрипторов**
**BRIEF — бинаризация патча вокруг ключевой точки:**
Для двух точек \( p \) и \( q \) в патче:
\[
f(p, q) =
\begin{cases} 
1, & \text{если } I(p) < I(q), \\ 
0, & \text{иначе}.
\end{cases}
\]
Здесь \( I(p) \) и \( I(q) \) — интенсивности пикселей \( p \) и \( q \).

Дескриптор \( d \) состоит из \( n \) таких сравнений:
\[
d = \{ f(p_1, q_1), f(p_2, q_2), \dots, f(p_n, q_n) \},
\]
где \( n = 256 \) (длина дескриптора).

---

### 4. **Сопоставление ключевых точек**
В коде:
```python
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
```

**Хамминговое расстояние** для сравнения бинарных дескрипторов:
\[
d_H(A, B) = \sum_{i=1}^n (A_i \oplus B_i),
\]
где:
- \( A \) и \( B \) — бинарные дескрипторы (256 бит),
- \( \oplus \) — XOR,
- \( d_H \) — количество различных бит между \( A \) и \( B \).

**Цель:**
Найти минимальное расстояние \( d_H \), чтобы определить совпадение.

s

### 5. **Визуализация совпадений**
В коде:
```python
final_img = cv2.drawMatches(main, queryKeypoints, example, trainKeypoints, matches[:20], None)
```

Используются координаты ключевых точек:
- Для каждого совпадения рисуется линия между соответствующими точками на изображениях \( A \) и \( B \).


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
\[
I_i > I_p + t \quad \text{или} \quad I_i < I_p - t,
\]
где:
- \( I_p \) — интенсивность центрального пикселя,
- \( I_i \) — интенсивности пикселей на окружности,
- \( t \) — пороговое значение.



### 3. **BRIEF для бинарных дескрипторов**
**BRIEF — бинаризация патча вокруг ключевой точки:**
Для двух точек \( p \) и \( q \) в патче:
\[
f(p, q) =
\begin{cases} 
1, & \text{если } I(p) < I(q), \\ 
0, & \text{иначе}.
\end{cases}
\]
Здесь \( I(p) \) и \( I(q) \) — интенсивности пикселей \( p \) и \( q \).

Дескриптор \( d \) состоит из \( n \) таких сравнений:
\[
d = \{ f(p_1, q_1), f(p_2, q_2), \dots, f(p_n, q_n) \},
\]
где \( n = 256 \) (длина дескриптора).


### 4. **Сопоставление ключевых точек**
В коде:
```python
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
```

**Хамминговое расстояние** для сравнения бинарных дескрипторов:
\[
d_H(A, B) = \sum_{i=1}^n (A_i \oplus B_i),
\]
где:
- \( A \) и \( B \) — бинарные дескрипторы (256 бит),
- \( \oplus \) — XOR,
- \( d_H \) — количество различных бит между \( A \) и \( B \).

**Цель:**
Найти минимальное расстояние \( d_H \), чтобы определить совпадение.



### 5. **Визуализация совпадений**
В коде:
```python
final_img = cv2.drawMatches(main, queryKeypoints, example, trainKeypoints, matches[:20], None)
```

Используются координаты ключевых точек:
- Для каждого совпадения рисуется линия между соответствующими точками на изображениях \( A \) и \( B \).



**Визуализация результатов:**
Результаты выводятся в сетке \( \text{rows} \times \text{cols} \), где каждая ячейка — это результат сопоставления двух изображений.




# README: ORB (Oriented FAST and Rotated BRIEF) Implementation

## Описание проекта

Этот проект реализует алгоритм ORB для поиска и сопоставления ключевых точек между изображениями. Программа обрабатывает пары изображений из двух директорий, определяет ключевые точки, создает дескрипторы, сопоставляет их и визуализирует результаты в едином выводе.



## Теоретическая база

### 1. **Ключевые точки на обоих изображениях**
В коде:
```python
queryKeypoints, queryDescriptors = orb.detectAndCompute(main_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(example_bw, None)
```



### 2. **FAST для обнаружения ключевых точек**
ORB использует алгоритм **FAST** для определения угловых точек.

**Условие ключевой точки:**
\[
I_i > I_p + t \quad \text{или} \quad I_i < I_p - t,
\]
где:
- \( I_p \) — интенсивность центрального пикселя,
- \( I_i \) — интенсивности пикселей на окружности,
- \( t \) — пороговое значение.



### 3. **BRIEF для бинарных дескрипторов**
BRIEF создает бинарный дескриптор на основе сравнений интенсивностей пикселей в патче вокруг ключевой точки.

**Бинаризация:**
Для двух точек \( p \) и \( q \) в патче:
\[
f(p, q) =
\begin{cases} 
1, & \text{если } I(p) < I(q), \\ 
0, & \text{иначе}.
\end{cases}
\]

Дескриптор \( d \) состоит из \( n \) таких сравнений:
\[
d = \{ f(p_1, q_1), f(p_2, q_2), \dots, f(p_n, q_n) \},
\]
где \( n = 256 \) (длина дескриптора).


### 4. **Сопоставление ключевых точек**
Для сопоставления дескрипторов используется метод Brute-Force Matcher:
```python
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
```

**Хамминговое расстояние для бинарных дескрипторов:**
\[
d_H(A, B) = \sum_{i=1}^n (A_i \oplus B_i),
\]
где:
- \( A \) и \( B \) — бинарные дескрипторы,
- \( \oplus \) — XOR,
- \( d_H \) — количество различных бит.



### 5. **Визуализация совпадений**
ORB визуализирует совпадения между изображениями:
```python
final_img = cv2.drawMatches(main, queryKeypoints, example, trainKeypoints, matches[:20], None)
```

Используются координаты ключевых точек, чтобы нарисовать линии между совпавшими точками.


## Визуализация результатов
Результаты выводятся в сетке \( \text{rows} \times \text{cols} \), где каждая ячейка отображает результат сопоставления двух изображений.

### Пример визуализации:
![Пример визуализации](Figure_1.png)



## Используемые источники

| №  | Источник                                                                                 | Описание                                   |
|----|------------------------------------------------------------------------------------------|-------------------------------------------|
| 1  | [Формулы и объяснение ORB](https://medium.com/@imhongxiaohui/explanation-of-orb-in-point-feature-extraction-1cdd9b82655a)       | Принцип работы алгоритма |
| 2  | [Статья про ORB](https://habr.com/ru/articles/414459/) | Принцип работы алгоритма    |


