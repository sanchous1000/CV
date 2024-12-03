import cv2
import os
import matplotlib.pyplot as plt
def ORB(main, example):
    main_bw = cv2.cvtColor(main,cv2.COLOR_BGR2GRAY)
    example_bw = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
    

    orb = cv2.ORB_create()

    queryKeypoints, queryDescriptors = orb.detectAndCompute(main_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(example_bw,None)


    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors,trainDescriptors)
    


    final_img = cv2.drawMatches(main, queryKeypoints, 
    example, trainKeypoints, matches[:20],None)
    

    final_img = cv2.resize(final_img, (1000,650))
    return final_img

    ''' cv2.imshow("Matches", final_img)
    cv2.waitKey(3000)'''

results = []
sample1 = os.listdir('samples1')
sample2 = os.listdir('samples2')
for i in sample1:
    main = cv2.imread(f'samples1/{i}')
    example = cv2.imread(f'samples2/{i}')
    result_img = ORB(main, example)
    results.append(result_img)

n = len(results) 
cols = 4  
rows = (n + cols - 1) // cols  

plt.figure(figsize=(25, 15))
for idx, result in enumerate(results):
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Сравнение {idx + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()