import math

import cv2
import numpy as np

kernel = np.array([(1, 1, 2, 2, 2, 1, 1), (1, 2, 2, 4, 2, 2, 1), (2, 2, 4, 8, 4, 2, 2), (2, 4, 8, 16, 8, 4, 2), (2, 2, 4, 8, 4, 2, 2),(1, 2, 2, 4, 2, 2, 1), (1, 1, 2, 2, 2, 1, 1)])
kn = (kernel.shape[0] - 1) // 2
km = (kernel.shape[1] - 1) // 2

kernelx = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
kxn = (kernelx.shape[0] - 1) // 2
kxm = (kernelx.shape[1] - 1) // 2

kernely = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
kyn = (kernely.shape[0] - 1) // 2
kym = (kernely.shape[1] - 1) // 2


def guass_blur(img):

    img = np.array(img,np.int64)
    imgn = img.shape[0]-1
    imgm = img.shape[1]-1
    cimg = np.zeros_like(img)

    for i in range(kn):
        cimg[i,:] = -1
        cimg[imgn-i,:] = -1
    for i in range(km):
        cimg[:,i] = -1
        cimg[:,imgm-i] = -1

    #print(cimg[4,:])

    for i in range(kn, imgn - kn+1):
        for j in range(km, imgm - km+1):
            cimg[i, j] = (kernel * img[i-kn:i+kn+1, j-km:j+km+1]).sum() // 140
            #print(cimg[i,j])
            #print(img[i-kn:i+kn+1, j-km:j+km+1])
            #print(i-kn,i+kn+1, j-km,j+km+1)
            #break
        #break
    return cimg


def sobel(gimg):

    gimg = np.array(gimg,np.int64)
    gimgn = gimg.shape[0] - 1
    gimgm = gimg.shape[1] - 1
    gximg = np.zeros_like(gimg)
    gyimg = np.zeros_like(gimg)
    gmimg = np.zeros_like(gimg)
    # gaimg = np.zeros_like(gimg,np.float64)
    gaimg = np.zeros_like(gimg)

    #print(gximg.dtype)

    for i in range(kn+kxn):
        gximg[i,:] = -1
        gximg[gimgn-i,:] = -1
        gyimg[i, :] = -1
        gyimg[gimgn-i, :] = -1
        gmimg[i, :] = -1
        gmimg[gimgn-i,:] = -1
        gaimg[i, :] = -1
        gaimg[gimgn - i, :] = -1
    for i in range(km+kxm):
        gximg[:,i] = -1
        gximg[:,gimgm-i] = -1
        gyimg[:, i] = -1
        gyimg[:, gimgm - i] = -1
        gmimg[:,i] = -1
        gmimg[:,gimgm-i] = -1
        gaimg[:, i] = -1
        gaimg[:, gimgm - i] = -1

    # print(gximg[0:7, 0:7])
    # print(gyimg[0:7, 0:7])
    # print(ggimg[0:7, 0:7])
    # print(gaimg[0:7, 0:7])


    for i in range(kn+kxn, gimgn - kn-kxn+1):
        for j in range(km+kxm, gimgm - km-kxm+1):
            gximg[i, j] = (kernelx * gimg[i-kxn:i+kxn+1, j-kxm:j+kxm+1]).sum()
            #print(gimg[i-kxn:i+kxn+1, j-kxm:j+kxm+1])
            #print(i-kxn,i+kxn+1, j-kxm,j+kxm+1)
            #print(gximg[i, j])

    for i in range(kn+kyn, gimgn - kn-kyn+1):
        for j in range(km+kym, gimgm - km-kym+1):
            gyimg[i, j] = (kernely * gimg[i-kyn:i+kyn+1, j-kym:j+kym+1]).sum()
            #print(gimg[i-kyn:i+kyn+1, j-kym:j+kym+1])
            #print(i-kyn,i+kyn+1, j-kym,j+kym+1)
            #print(gyimg[i, j])

    for i in range(kn+kyn, gimgn - kn-kyn+1):
        for j in range(km+kym, gimgm - km-kym+1):
            gmimg[i, j] = math.sqrt((gximg[i,j] * gximg[i,j]) + (gyimg[i,j] * gyimg[i,j]))

    # print(gximg[156:160, 4:7])
    # print(gyimg[156:160, 4:7])
    # print(ggimg[156:160, 4:7])

    for i in range(kn + kyn, gimgn - kn - kyn + 1):
        for j in range(km + kym, gimgm - km - kym + 1):
            if gximg[i,j] != 0 :
                gaimg[i, j] = np.degrees(np.arctan((gyimg[i,j]/gximg[i,j])))
            # elif gximg[i,j] == 0 and gyimg[i,j] > 0:
            elif gximg[i, j] == 0:
                gaimg[i,j] = 90

    # print(gaimg[178])
    # print(gaimg.min())
    # print(gaimg.max())

    gximg = abs(gximg//4)
    gyimg = abs(gyimg//4)
    gmimg = abs(gmimg//4)

    return gximg,gyimg,gmimg,gaimg

    #print(gximg.max(),gyimg.max())
    #print(gximg.min(),gyimg.min())


def quadrant(i):
    pass


def nonmaxima_supress(gmimg,gaimg):

    gmimgn = gmimg.shape[0] - 1
    gmimgm = gmimg.shape[1] - 1
    nmsimg = np.zeros_like(gmimg)

    # for i in range(kn+kyn, gmimgn - kn-kyn+1):
    #     for j in range(km+kym, gmimgm - km-kym+1):
    #         i = quadrant(gaimg[i,j])
    #         if i == 0 and gmimg[]
    #         nmsimg[i, j] =


if __name__ == "__main__":

    #imgpath = input("Enter the path to the image:")
    #img = cv2.imread(imgpath,0)
    # img = cv2.imread('/Users/umar/Documents/courses/cv_6643/project1/Houses-225.bmp',0)
    img = cv2.imread('/Users/umar/Documents/courses/cv_6643/project1/Zebra-crossing-1.bmp', 0)
    gimg = guass_blur(img)
    #print(gimg)
    """cv2.imshow('guass', gimg)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):"""
    cv2.imwrite('gs_img.png',gimg)
    gximg,gyimg,gmimg,gaimg = sobel(gimg)
    cv2.imwrite('gx_img.png', gximg)
    cv2.imwrite('gy_img.png', gyimg)
    cv2.imwrite('gm_img.png', gmimg)
    cv2.imwrite('ga_img.png', gaimg)

    # nmsimg = nonmaxima_supress(gmimg,gaimg)