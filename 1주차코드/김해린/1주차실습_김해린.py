import cv2
import numpy as np
import pdb
#pdb.set_trace()

cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while True:
    ret_val, img=cam.read()
    cv2.imshow("Cam Viewer",img)
    key=cv2.waitKey(1)
    if key == 27:#esc 누르면 정지
        break

    img1=cv2.imread("C:\\opencv-2-4-13-6\\correspondence_problem.jpg")
    
    sift=cv2.xfeatures2d.SIFT_create()

    kp1=sift.detect(img1,None)
    kp2=sift.detect(img,None)

    kp1,des1=sift.compute(img1,kp1)
    kp2,des2=sift.compute(img,kp2)

    FlANN_INDEX_KDTREE=0
    index_params=dict(algorithm=FlANN_INDEX_KDTREE,trees=5)
    
    search_params=dict(checks=50)

    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(des1,des2,k=2)

    res=None
    good=[]

    for m,n in matches:
        if m.distance<0.5*n.distance:
            good.append(m)
    print(len(good))
    res=cv2.drawMatches(img1,kp1,img,kp2,good,res,flags=0)

    cv2.imshow('Feature Matching',res)

    #pdb.set_trace()
    MIN_MATCH_COUNT=4
    if len(good)>MIN_MATCH_COUNT:
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good])

        M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
        print(M)
        matchesMask=mask.ravel().tolist()

        h,w,channel=img1.shape
        #pdb.set_trace()
        pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst=cv2.perspectiveTransform(pts,M)
        img=cv2.polylines(img,[np.int32(dst)],True,0,3,cv2.LINE_AA)
    else:
        print("not enough matches",len(good))
        matchesMask=None

    img3=cv2.drawMatches(img1,kp1,img,kp2,good,None,(0,0,0),None,matchesMask,2)
    cv2.imshow('imgbox',img3)
