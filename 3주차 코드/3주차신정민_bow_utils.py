import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import scipy.cluster.vq as vq

def load(data):
    if data == 'train':
        print('loading train')
        path = "./Caltech101"
        folder_list = os.listdir(path)#jpg파일이 들어있는 폴더들의 리스트
        sift = cv2.xfeatures2d_SIFT.create()
        print("folder_list: {}".format(folder_list))
        image_feature_list = []
        y_train = []

        for i in range(len(folder_list)):
            file_list = os.listdir(path + "/" + folder_list[i])
            print(folder_list[i])

            for j in range(30):
                gray = cv2.imread(path + "/" + folder_list[i] + "/" + file_list[j], cv2.IMREAD_GRAYSCALE)
                kp = sift.detect(gray)
                _,des = sift.compute(gray, kp, gray)  # _는 다신 사용하지 않을 변수를 쓸 때 사용
                image_feature_list.append(des)

                _, label = folder_list[i].split('.')  # 라벨을 구하는 과정
                y_train.append(label)

        image_feature_list, y_test = check_None(image_feature_list, y_train)
        # pickle.dump(image_feature_list, open("./gogo/des_train.npy", 'wb'))
        # pickle.dump(y_train, open("./gogo/label_train.npy", 'wb'))

        return image_feature_list, y_train

    elif data == 'test':
        print('loading test')
        path = "./Caltech101"
        folder_list = os.listdir(path)#jpg파일이 들어있는 폴더들의 리스트
        sift = cv2.xfeatures2d_SIFT.create()
        print("folder_list: {}".format(folder_list))
        image_feature_list = []
        y_test = []

        for i in range(len(folder_list)):
            file_list = os.listdir(path + "/" + folder_list[i])
            print(folder_list[i])

            for j in range(30,len(file_list)):
                gray = cv2.imread(path + "/" + folder_list[i] + "/" + file_list[j], cv2.IMREAD_GRAYSCALE)
                kp = sift.detect(gray)
                _,des = sift.compute(gray, kp, gray)  # _는 다신 사용하지 않을 변수를 쓸 때 사용
                image_feature_list.append(des)

                _, label = folder_list[i].split('.')  # 라벨을 구하는 과정
                y_test.append(label)

        image_feature_list, y_test = check_None(image_feature_list, y_test)
        # pickle.dump(image_feature_list, open("./gogo/des_test.npy", 'wb'))
        # pickle.dump(y_test, open("./gogo/label_test.npy", 'wb'))

        return image_feature_list, y_test


def check_None(x,y):
    find_None = []
    for i in range(len(x)):
        if x[i] is None:
            print(i)
            find_None.append(i)#None이 몇번째 리스트에 있는지를 저장

    for i in find_None:
        del x[i]
        del y[i]#None이 있을경우 del해줌

    print(np.array(y).shape)
    print(np.array(x).shape)#test용 des 및 label

    return x, y

def extend_des(x):
    list_des=[]
    for i in range(len(x)):
        if x[i] is not None:
            list_des.extend(x[i])
        else:
            print(i)
    #
    #
    # print(len(list_des))
    # print(np.array(list_des).shape)##sift를 통해 구한 des를 하나하나씩 extend하는 과정 갯수 1380891개
    # pickle.dump(list_des, open("./gogo/extend_des_train.npy", 'wb'))
    return list_des

def find_centroid(list_des):

    kmeans_center = KMeans(n_clusters=200, n_jobs=-2, verbose=1, algorithm='elkan')
    kmeans_center.fit(list_des)  # fit해야 군집 완성

    codebook = kmeans_center.cluster_centers_.squeeze()

    # print("finish_method")
    return codebook

def make_codebook(list_des,codebook):
    code, _ = vq.vq(list_des,codebook)#codeword를 만드는 과정
    word_his, bin_edges = np.histogram(code, bins=range(codebook.shape[0]+1), normed=True)
    return word_his

def classifier(x_train,y_train,x_test=None,y_test=None):
    # clf = SVC(gamma='auto', cache_size=12000, max_iter=-1)
    # print("Training the data set...")
    # clf = clf.fit(x_train, y_train)
    # print("Training Completed")
    #
    # result = clf.predict(x_test)
    #
    # score = accuracy_score(y_test,result)
    # print(score)


    # # x_train,x_test,y_train,y_test = train_test_split(x_train, y_train ,test_size=0.2,random_state=6)
    #
    #
    # estimator = SVC(kernel='rbf',C=100.0,gamma=100.0)
    estimator = SVC(kernel='linear', C=10.0)

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_train)
    score = accuracy_score(y_train, y_predict)
    print(score)  # 1.0

    y_predict = estimator.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    print(score)  # 1.0

    # min_on_training = x_train.min(axis=0)
    #
    # range_on_training = (x_train - min_on_training).max(axis=0)
    #
    # x_train_scaled = (x_train - min_on_training) / range_on_training
    #
    # print("특성별 최소 값\n{}".format(x_train_scaled.min(axis=0)))
    # print("특성별 최대 값\n {}".format(x_train_scaled.max(axis=0)))
    #
    # x_test_scaled = (x_test - min_on_training) / range_on_training
    #
    # svc = SVC(C=10)
    # svc.fit(x_train,y_train)
    #
    # print("훈련 세트 정확도: {:.2f}".format(svc.score(x_train, y_train)))
    # print("테스트 세트 정확도: {:.2f}".format(svc.score(x_test, y_test)))







    # sc = StandardScaler()
    # sc.fit(x_train)
    # x_train_std = sc.transform(x_train)
    # x_test_std = sc.transform(x_test)
    #
    # ml = SVC(kernel='rbf', C=10.0, gamma=0.10, random_state=0)
    # ml.fit(x_train, y_train)
    # y_pred = ml.predict(x_test_std)
    # print("총 테스트 개수:%d, 오류개수: %d" %(len(y_test),(y_test != y_pred).sum()))
    # print('정확도: %.2f' %accuracy_score(y_test,y_pred))
    # x_combined_std = np.vstack(x_train_std, x_test_std)
    # y_combined_std = np.hstack((y_train,y_test))
    # plot_decision_regions(x_combined_std, y_combined_std, classifier=ml, title='scikit-learn svm kernel = "rbf"')


