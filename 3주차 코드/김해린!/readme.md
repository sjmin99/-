ex) 딥러닝: Feature extraction + classifer

+ Bag of Words + SVM : bag of keypoints기반의 이미지 분류 

> 시각적 분류를 할 때, 이미지 패치당(부분적으로) 특징을 찾아서 분류를 하는 방법입니다.

cifar10 데이터셋을 바탕으로 Sift알고리즘으로 피쳐 추출 기술을 한 후, kmeans라는 최적화 함수를 사용하여 특징을 찾고  각 이미지당 히스토그램을 쌓아서 분포를 통해 SVM 분류기를 통해 분류합니다. 



```
import numpy as np
import cv2
from sklearn.cluster import KMeans
import scipy.cluster.vq as vq
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

VOC_SIZE = 200
DSIFT_STEP_SIZE = 4


def load_cifar10_data(dataset):
    if dataset == 'train':
        with open('./cifar10/train/train.txt', 'r') as f:
            paths = f.readlines()
    if dataset == 'test':
        with open('./cifar10/test/test.txt', 'r') as f:
            paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        path, label = each.split(' ')
        img = cv2.imread(path)
        x.append(img)
        y.append(label)
    return [x, y]


def extract_sift_descriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # keypoints,descriptors=sift.detectAndCompute(gray,None)
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
                 for y in range(0, gray.shape[0], disft_step_size)
                 for x in range(0, gray.shape[1], disft_step_size)]
    keypoints, descriptors = sift.compute(gray, keypoints)
    return descriptors


def build_codebook(X, voc_size):
    # features=np.vstack(descriptor for descriptor in X)
    features = np.vstack((descriptor for descriptor in X)).astype(np.float32)
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2)
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_.squeeze()
    return codebook


def input_vector_encoder(feature, codebook):
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist


if __name__ == '__main__':
    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    print("SIFT feature extraction")
    x_train = [extract_sift_descriptors(img) for img in x_train]
    x_test = [extract_sift_descriptors(img) for img in x_test]
    x_train = [each for each in zip(x_train, y_train) if not each[0] is None]
    x_train, y_train = zip(*x_train)

    print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
    print("Codebook Size: {:d}".format(VOC_SIZE))

    x_test = [each for each in zip(x_test, y_test) if not each[0] is None]
    x_test, y_test = zip(*x_test)

    print("Building the codebook, it will take some time")
    codebook = build_codebook(x_train, voc_size=VOC_SIZE)

    print("Bag of words encoding")
    x_train = [input_vector_encoder(x, codebook) for x in x_train]
    x_train = np.asarray(x_train)
    x_test = [input_vector_encoder(each, codebook) for each in x_test]
    x_test = np.asarray(x_test)

    svc.fit(x_train, y_train)

    print("훈련 세트 정확도: {:.3f}".format(svc.score(x_train, y_train)))
    print("테스트 세트 정확도: {:.3f}".format(svc.score(x_test, y_test)))
```

+ 참고 코드:  https://github.com/CyrusChiu/Image-recognition
+ 참고 논문:  "Visual Categorization with Bag of Keypoints"

--------------------------------------------------------------------------------------------

+ Experiment(K=100일때, Dense Sift)
![image](https://user-images.githubusercontent.com/44723287/63140164-e29b7680-c01b-11e9-9eab-194d90e5cbb5.png)
