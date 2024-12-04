import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def ORB(main, example, am_match = 10):
    main_bw = cv2.cvtColor(main,cv2.COLOR_BGR2GRAY)
    example_bw = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
    

    orb = cv2.ORB_create()

    #Находим ключевые точки и дескрипторы (локальные свойсвта изображения )
    main_keypoints, main_descriptors = orb.detectAndCompute(main_bw,None)
    example_keypoints, example_descriptors = orb.detectAndCompute(example_bw,None)

    #Далее надо найти свпадения между ними на основе дескрипторов
    matcher = cv2.BFMatcher() #cv.NORM_HAMMING, crossCheck=True
    matches = matcher.match(main_descriptors,example_descriptors)
    matches = sorted(matches, key=lambda x: x.distance) #сортируем точки по близости
    good_matches = matches[:am_match]  
    points1 = np.array([main_keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    points2 = np.array([example_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32)


    #Визуализация результата при помощи матрицы преобразования
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    h, w, _ = main.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H) #изменение перспективы ищображения

    image2_with_box = cv2.polylines(example.copy(), [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)

    return image2_with_box
    

    

results = []

v = os.listdir('main')
sample = os.listdir('samples1')
for i in sample:
    if 'mb' not in i:
        main = cv2.imread(f'main/{i}')
        example = cv2.imread(f'samples1/{i}')
        result_img = ORB(main, example)
        results.append((main, result_img)) 
    else:
        print(i)
        main = cv2.imread(r'C:\Users\Aleks\Documents\CV\HW2_part2\main\main_mb.jpg')
        example = cv2.imread(f'samples1/{i}')
        result_img = ORB(main, example, am_match = 10)
        results.append((main, result_img)) 
    



def plots(results):
    for idx, (main, result_img) in enumerate(results):
        # Convert images to RGB for correct display
        main_image_rgb = cv2.cvtColor(main, cv2.COLOR_BGR2RGB)
        result_image_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Plot the main image in the first row
        axes[0, idx].imshow(main_image_rgb)
        axes[0, idx].set_title(f"Main Image {idx + 1}")
        axes[0, idx].axis("off")

        # Plot the result image in the second row
        axes[1, idx].imshow(result_image_rgb)
        axes[1, idx].set_title(f"Result Image {idx + 1}")
        axes[1, idx].axis("off")

   

    # Adjust layout for better visibility
    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots( 2,len(results[:5]), figsize=(15, 10))
plots(results[5:10])