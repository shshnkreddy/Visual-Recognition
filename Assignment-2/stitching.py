import cv2   
import numpy as np

scale = 0.6
images = []
images.append(cv2.imread(r'/home/shshnkreddy/Downloads/Pool1.jpeg'))
images.append(cv2.imread(r'/home/shshnkreddy/Downloads/Pool2.jpeg'))
images.append(cv2.imread(r'/home/shshnkreddy/Downloads/Pool3.jpeg'))
images.append(cv2.imread(r'/home/shshnkreddy/Downloads/Pool4.jpeg'))

def Scale(image, scale):
    return cv2.resize(image,(int(scale*image.shape[1]),int(scale*image.shape[0])),interpolation = cv2.INTER_AREA)

for i in range(len(images)):
	images[i] = Scale(images[i],scale)

def trim(frame):
    #crop top
	if not np.sum(frame[0]):
		return trim(frame[1:])
    #crop top
	if not np.sum(frame[-1]):
		return trim(frame[:-2])
    #crop top
	if not np.sum(frame[:,0]):
		return trim(frame[:,1:])
    #crop top
	if not np.sum(frame[:,-1]):
		return trim(frame[:,:-2])
	return frame

def stitch(img1,img2,method):
	if(method == 'surf'):
		surf = cv2.xfeatures2d.SURF_create()
		kp1, des1 = surf.detectAndCompute(img1,None)
		kp2, des2 = surf.detectAndCompute(img2,None) 	
	else:	
		sift = cv2.xfeatures2d.SIFT_create()
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None) 

	# interest_points = np.concatenate((cv2.drawKeypoints(img1,kp1,None),cv2.drawKeypoints(img2,kp1,None)), axis = 1)
	# cv2.imshow('Interest Points', interest_points)
	# cv2.waitKey(0)
	
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	match = cv2.FlannBasedMatcher(index_params, search_params)
	matches = match.knnMatch(des1,des2,k=2)

	good = []
	for m,n in matches:
	    if m.distance < 0.3*n.distance:
	        good.append(m)

	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	# cv2.imshow("original_image_drawMatches.jpg", img3)
	# cv2.waitKey(0)
	
	MIN_MATCH_COUNT = 10
	if len(good) > MIN_MATCH_COUNT:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

	    h,w,ch = img1.shape
	    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    dst = cv2.perspectiveTransform(pts,M)

	    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	    # cv2.imshow("original_image_overlapping.jpg", img2)
	    # cv2.waitKey(0)
	else:
		print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

	dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
	dst[0:img2.shape[0],0:img2.shape[1]] = img2
	# cv2.imshow("original_image_stitched.jpg", dst)
	# cv2.waitKey(0)
	# cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
	# cv2.waitKey(0)
	return trim(dst)
	


# res = stitch(images[0].copy(),images[1].copy(), 'sift')
# horizontal = np.concatenate((image1, image2), axis=1)
# cv2.imshow('Original Images', horizontal);
# horizontal = images[0]
# for i in range(1,len(images)):
# 	horizontal = np.concatenate((horizontal,images[i]), axis = 1)

# horizontal = Scale(horizontal, scale)
res = stitch(images[0],images[1], 'surf')
for i in range(2,len(images)):
	res = stitch(res,images[i],'surf')

cv2.imshow('Panorama', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
