import cv2
import numpy as np


cam = cv2.VideoCapture(0)  # 비디오 촬영
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # size 지정
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # size 지정
def featureMatching(frame):#비디오
    img1 = frame#카메라 영상
    img2 = cv2.imread('D:/week1-c.png', cv2.IMREAD_COLOR)#영상과 비교할 사진

    sift = cv2.xfeatures2d.SIFT_create()#SIFT 사용 준비

    kp1=sift.detect(img1,None)#sift를 이용하여 키포인트 검출
    kp2=sift.detect(img2,None)
    kp1, des1 = sift.compute(img1, kp1)#키포인트에서 디스크립트 계산
    kp2, des2 = sift.compute(img2, kp2)#

    #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)#BF매칭 이용
    #matches = bf.match(des1, des2)#

    #matches = sorted(matches, key=lambda x:x.distance)
    res = None
    #FLANN 매칭 사용
    FLANN_INDEX_KDTREE = 0#
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)#특성매칭 반복 횟수

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)#2번째로 가까운매칭 결과까지 리턴,즉 matches는 1순위 2순위 매칭 결과가 멤버인 리스트



    good = []
    for m, n in matches:
        if m.distance < 0.5* n.distance:
            good.append(m)#matches에서 1순위 매칭 결과가 2순위 매칭 결과의 0.5보다 더 가까운 값을 취함
    print(len(good))
    res = cv2.drawMatches(img1, kp1, img2, kp2, good, res, flags=0)

    cv2.imshow('Matching', res)#res 화면 출력
    return good
def box(frame):

    img1=frame
    img2=cv2.imread('D:/week1-c.png',cv2.IMREAD_COLOR)#이미지 컬러
    #sift 객체 선언
    sift=cv2.xfeatures2d.SIFT_create()
    good=featureMatching(frame)
    kp1=sift.detect(img1,None)
    kp2=sift.detect(img2,None)

    MIN_MATCH_COUNT=4

    if len(good)>MIN_MATCH_COUNT:
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good])

        M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        print(M)
        matchesMask=mask.ravel().tolist()
        h,w,channel=img1.shape
        pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst=cv2.perspectiveTransform(pts,M)
        img1=cv2.polylines(img1,[np.int32(dst)],True,0,3,cv2.LINE_AA)#외곽선 그리기
    else:
        print("not enough matches",len(good))
        matchesMask=None
    img3=cv2.drawMatches(img1,kp1,img2,kp2,good,None,(0,0,0),None,matchesMask,2)
    return cv2.imshow("Matching",img3)#이미지출력


while(1):

    ret, frame=cam.read()
    cv2.imshow('cam',frame)

    box(frame)

    k=cv2.waitKey(100)
    if k==27:
        break
#cv2.xfeatures2d.SIFT_create()->SIFT의 키포인트 디스크립터들을 계산하는 함수를 제공
#cv2.dect(grayimg)->grayimg에서 키포인트를 검출하여 리턴
#cv2.compute(keypoints)->키포인트에서 디스크립터를 계산한 후 키포인트와 디스크립터를 리턴
#cv2.detectAndCompute(grayimg)->grayimg에서 키포인트와 디스크립터를 한번에 표시하고 계산
#cv2.drawKeypoints(grayimg,keypoints,outing)->grayimg에 키포인트들을 outimg에 표시

cv2.release()
cv2.destroyAllWindows()


