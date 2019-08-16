import numpy as np
from bow_utils import load
import pickle
from sklearn.cluster import KMeans
from bow_utils import find_centroid
from bow_utils import make_codebook
from bow_utils import classifier
from bow_utils import extend_des
#

x_train, y_train = load('train')
x_test, y_test = load('test')
# print('finish')

# x_train = pickle.load(open("./gogo/des_train.npy","rb"))
# y_train = pickle.load(open("./gogo/label_train.npy","rb"))#train용 des 및 label
# print(np.array(y_train).shape)
# print(np.array(x_train).shape)

# x_test = pickle.load(open("./gogo/des_test.npy", "rb"))
# y_test = pickle.load(open("./gogo/label_test.npy", "rb"))

list_des_train = extend_des(x_train)
list_des_test = extend_des(x_test)

kmeans_center = find_centroid(list_des_train)#kmenas로 군집화하는 함수

# kmeans_center = pickle.load(open("./gogo/centroid_200.npy","rb"))
# print(kmeans_center.shape)##kmeans를 통해 군집화
#
# #if(traing)
word_hist = []
word_hist = [make_codebook(x, kmeans_center) for x in x_train]#traing 히스토그램
word_hist = np.asarray(word_hist)
# pickle.dump(word_hist, open("./gogo/word_hist_train_200.npy", 'wb'))
# print(np.array(word_hist).shape)##히스토그램을 만드는 과정들

#elif(testing)
word_hist = []
word_hist = [make_codebook(x, kmeans_center) for x in x_test]#traing 히스토그램
word_hist = np.asarray(word_hist)
# pickle.dump(word_hist, open("./gogo/word_hist_test_200.npy", 'wb'))
# print(np.array(word_hist).shape)##히스토그램을 만드는 과정들

# x_train = pickle.load(open("./gogo/word_hist_train_200.npy", "rb"))
# x_train = np.asarray(x_train)
# x_test = pickle.load(open("./gogo/word_hist_test_200.npy", "rb"))
# x_test = np.asarray(x_test)
#
classifier(x_train,y_train,x_test,y_test)