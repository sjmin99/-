import numpy as np
import cv2
imgname = "D:/week1-c.png"       # 컴퓨터 이미지1
imgname2 = "D:/week1-b.jpg" # 컴퓨터 이미지2

MIN_MATCH_COUNT = 4

orb = cv2.ORB_create()# orb 객체 초기화
img1 = cv2.imread(imgname)#사진 불러오기
img2 = cv2.imread(imgname2)#사진 불러오기

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#기본 BGR 컬러 gray로 변경
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)#기본 BGR 컬러 gray로 변경

kpts1, descs1 = orb.detectAndCompute(gray1,None)#키포인트와 디스크립터 계산.
kpts2, descs2 = orb.detectAndCompute(gray2,None)#키포인트와 디스크립터 계산.

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)#bf매칭 객체를 cv2.NORM_HAMMING으로 crossCheck값을 true로 설정
matches = bf.match(descs1, descs2)#두이미지 디스크립터 매칭 결과 저장
dmatches = sorted(matches, key = lambda x:x.distance)#두 이미지 특성 포인트들을 가장 일치하는 순서대로 정렬

#homopraphy:한평면 위에 다른 평면 투영시, 투영된 대응점들 사이에서의 변환관계
src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)#투영변환

img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)#이미지에 외곽선을 그림

res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)#일치되는 특성 포인트만 표시->flasg=2,0인경우 특성 포인트 모두 화면 표시

cv2.imshow("orb_match", res)#이미지 보여주기

cv2.waitKey()
cv2.destroyAllWindows()#모든창 닫기
