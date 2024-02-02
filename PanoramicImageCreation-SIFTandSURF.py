import numpy as np
import cv2 as cv


def panorama(filename1,filename2):


    # SURF

 #detector_descriptor = cv.xfeatures2d_SURF.create(5000)


    # SIFT
 detector_descriptor = cv.xfeatures2d_SIFT.create(1000)
 cv.namedWindow('main1')
 img1 = cv.imread(filename1)
 cv.imshow('main1', img1)
 cv.waitKey(0)

 kp1 = detector_descriptor.detect(img1)
 desc1 = detector_descriptor.compute(img1, kp1)
 #draw the keypoints in img1
    #grayimg1 = cv.imread(filename1, cv.IMREAD_GRAYSCALE)
    # kpimg1 = cv.drawKeypoints(grayimg1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imwrite('keypoints1.jpg', kpimg1)
    # cv.namedWindow('keypoints1')
    # cv.imshow('keypoints1', kpimg1)
    # cv.waitKey(0)

 img2 = cv.imread(filename2)
 cv.namedWindow('main2')
 cv.imshow('main2', img2)
 cv.waitKey(0)

 kp2 = detector_descriptor.detect(img2)
 desc2 = detector_descriptor.compute(img2, kp2)
 #uncomment if you want to draw the keypoints in img2
 # grayimg2 = cv.imread(filename2, cv.IMREAD_GRAYSCALE)
 # kpimg2 = cv.drawKeypoints(grayimg2, kp1, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 # cv.imwrite('keypoints2.jpg', kpimg2)
 # cv.namedWindow('keypoints2')
 # cv.imshow('keypoints2', kpimg2)
 # cv.waitKey(0)

 def match(d1, d2):
     n1 = d1.shape[0]#img1 keypoints
     n2 = d2.shape[0]#img2 keypoints

     matches = []
     for i in range(n1):
         fv = d1[i, :]#for every keypoint of img1
         diff = d2 - fv #calculate the manhatan distance from every keypoint of img2
         diff = np.abs(diff)
         distances = np.sum(diff, axis=1)#sums all the values of every row

         i2 = np.argmin(distances)#position of min distance
         mindist = distances[i2] #min disnance



         matches.append(cv.DMatch(i, i2, mindist))

     return matches

 matches_1 = match(desc1[1], desc2[1])
 dimg1 = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches_1, None)
 cv.namedWindow('matches1')
 cv.imshow('matches1', dimg1)
 cv.imwrite('dimg1.png', dimg1)
 cv.waitKey(0)


 matches_2 = match(desc2[1], desc1[1])


 #create the cross check
 arr1=[]
 arr2=[]
 i=0
 matchesfinal=[]
 for x in matches_1:
     arr1.append([x.queryIdx,x.trainIdx])
     for y in matches_2:
         arr2.append([y.queryIdx,y.trainIdx])

         if x.queryIdx==y.trainIdx and x.trainIdx==y.queryIdx:
             matchesfinal.append(matches_1[i])
     i=i+1
 matchesfinal=np.array(matchesfinal)
 dimg = cv.drawMatches(img1, desc1[0], img2, desc2[0], matchesfinal, None)
 cv.namedWindow('matches')
 cv.imshow('matches', dimg)
 cv.imwrite('dimg.png',dimg)
 cv.waitKey(0)

 img_pt1 = []
 img_pt2 = []
 for x in matchesfinal:
     img_pt1.append(kp1[x.queryIdx].pt)
     img_pt2.append(kp2[x.trainIdx].pt)
 img_pt1 = np.array(img_pt1)
 img_pt2 = np.array(img_pt2)

 M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)


 img3 = cv.warpPerspective(img2, M, (img1.shape[1]+3000, img1.shape[0]+1000))
 img3[0: img1.shape[0], 0: img1.shape[1]] = img1

 cv.namedWindow('PANORAMA')
 cv.imshow('PANORAMA', img3)
 cv.waitKey(0)
 return img3


#make a function for croping the black borders of the panorama images
def cropborders(filename):
    img=cv.imread(filename)
    grayimg = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # create the binary img
    _, binary = cv.threshold(grayimg, 1, 255, cv.THRESH_BINARY)
    cv.namedWindow('binaryimg')
    cv.imshow('binaryimg', binary)
    cv.waitKey(0)
    num_cc, labeled, stats, centroids = cv.connectedComponentsWithStats(binary)
    x0 = stats[1, cv.CC_STAT_LEFT]
    y0 = stats[1, cv.CC_STAT_TOP]
    w = stats[1, cv.CC_STAT_WIDTH]
    h = stats[1, cv.CC_STAT_HEIGHT]
    crop=img[y0:y0+h,x0:x0+w-15]
    cv.namedWindow('panoramacroped')
    cv.imshow('panoramacroped', crop)
    cv.waitKey(0)
    return crop


#uncomment if you want to check my images

# panorama1=panorama('xanthi1.png','xanthi2.png')
# cv.imwrite('panorama1.png',panorama1)
# crop1=cropborders('panorama1.png')
# cv.imwrite('panorama1croped.png', crop1)
#
# panorama2=panorama('xanthi3.png','xanthi4.png')
# cv.imwrite('panorama2.png',panorama2)
# crop2=cropborders('panorama2.png')
# cv.imwrite('panorama2croped.png', crop2)

panorama1=panorama('rio-01.png','rio-02.png')
cv.imwrite('panorama1.png',panorama1)
crop1=cropborders('panorama1.png')
cv.imwrite('panorama1croped.png', crop1)

panorama2=panorama('rio-03.png','rio-04.png')
cv.imwrite('panorama2.png',panorama2)
crop2=cropborders('panorama2.png')
cv.imwrite('panorama2croped.png', crop2)

panoramafinal=panorama('panorama1croped.png','panorama2croped.png')
cv.imwrite('panoramafinal.png',panoramafinal)
crop3=cropborders('panoramafinal.png')
cv.imwrite('panoramafinalcroped.png', crop3)
