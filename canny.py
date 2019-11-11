import math
import os.path
import cv2
import numpy as np

#Guassian kernel

kernel = np.array([(1, 1, 2, 2, 2, 1, 1), (1, 2, 2, 4, 2, 2, 1), (2, 2, 4, 8, 4, 2, 2), (2, 4, 8, 16, 8, 4, 2), (2, 2, 4, 8, 4, 2, 2),(1, 2, 2, 4, 2, 2, 1), (1, 1, 2, 2, 2, 1, 1)])
kn = (kernel.shape[0] - 1) // 2
km = (kernel.shape[1] - 1) // 2

#Sobel gx operator

kernelx = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
kxn = (kernelx.shape[0] - 1) // 2
kxm = (kernelx.shape[1] - 1) // 2

#Sobel gy operator

kernely = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
kyn = (kernely.shape[0] - 1) // 2
kym = (kernely.shape[1] - 1) // 2

#Guassian smoothing with above given kernel

def guass_blur(img):

    img = np.array(img,np.float32)
    imgn = img.shape[0]-1
    imgm = img.shape[1]-1
    gimg = np.zeros_like(img)

    #Making pixels that go out of kernel as undefined(-1)

    for i in range(kn):
        gimg[i,:] = -1
        gimg[imgn-i,:] = -1
    for i in range(km):
        gimg[:,i] = -1
        gimg[:,imgm-i] = -1

    #Convolution and Normalisation with 140

    for i in range(kn, imgn - kn+1):
        for j in range(km, imgm - km+1):
            gimg[i, j] = (kernel * img[i-kn:i+kn+1, j-km:j+km+1]).sum() / 140

    return gimg


#Calculating gradient magnitude and gradient angle

def sobel(gimg):

    gimgn = gimg.shape[0] - 1
    gimgm = gimg.shape[1] - 1
    gximg = np.zeros_like(gimg)
    gyimg = np.zeros_like(gimg)
    gmimg = np.zeros_like(gimg)
    gaimg = np.zeros_like(gimg)

    #Making pixels that go out of kernel as undefined(-1)

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
        gyimg[:, gimgm-i] = -1
        gmimg[:,i] = -1
        gmimg[:,gimgm-i] = -1
        gaimg[:, i] = -1
        gaimg[:, gimgm-i] = -1

    #Convolution for x-gradient

    for i in range(kn+kxn, gimgn - kn-kxn+1):
        for j in range(km+kxm, gimgm - km-kxm+1):
            gximg[i, j] = (kernelx * gimg[i-kxn:i+kxn+1, j-kxm:j+kxm+1]).sum()

    #Convolution for y-gradient

    for i in range(kn+kyn, gimgn - kn-kyn+1):
        for j in range(km+kym, gimgm - km-kym+1):
            gyimg[i, j] = (kernely * gimg[i-kyn:i+kyn+1, j-kym:j+kym+1]).sum()

    #Gradient magnitude calcultaion

    for i in range(kn+kyn, gimgn - kn-kyn+1):
        for j in range(km+kym, gimgm - km-kym+1):
            gmimg[i, j] = math.sqrt((gximg[i,j] * gximg[i,j]) + (gyimg[i,j] * gyimg[i,j]))

    #Gradient angle calculation

    for i in range(kn + kyn, gimgn - kn-kyn+1):
        for j in range(km + kym, gimgm - km-kym+1):
            if gximg[i,j] != 0 :
                gaimg[i,j] = np.degrees(np.arctan((gyimg[i,j]/gximg[i,j])))
                if gaimg[i,j] < 0:
                    gaimg[i,j] += 360
            else:
                if gyimg[i,j] < 0:
                    gaimg[i,j] = 270
                else:
                    gaimg[i,j] = 90

    return gximg, gyimg, gmimg, gaimg

def nonmaxima_supress(gmimg,gaimg):

    gmimgn = gmimg.shape[0] - 1
    gmimgm = gmimg.shape[1] - 1
    nmsimg = np.zeros_like(gmimg)

    #Making pixels that go out of kernel as undefined(-1)

    for i in range(kn+kxn):
        nmsimg[i,:] = -1
        nmsimg[gmimgn-i,:] = -1
    for i in range(km+kxm):
        nmsimg[:,i] = -1
        nmsimg[:,gmimgm-i] = -1

    #Non-maxima supression based on quadrant values

    for i in range(kn+kyn, gmimgn - kn-kyn+1):
        for j in range(km+kym, gmimgm - km-kym+1):
            if (0 <= gaimg[i,j] < 22.5) or (157.5 <= gaimg[i,j] <= 202.5) or (337.5 <= gaimg[i,j] <=360):
                x = gmimg[i,j-1]
                y = gmimg[i,j+1]
            elif (22.5 <= gaimg[i,j] < 67.5) or (202.5 <= gaimg[i,j] < 247.5):
                x = gmimg[i-1,j+1]
                y = gmimg[i+1,j-1]
            elif (67.5 <= gaimg[i,j] < 112.5) or (247.5 <= gaimg[i,j] < 292.5):
                x = gmimg[i-1,j]
                y = gmimg[i+1,j]
            elif (112.5 <= gaimg[i,j] < 157.5) or (292.5 <= gaimg[i,j] < 337.5):
                x = gmimg[i-1,j-1]
                y = gmimg[i+1,j+1]
            if (gmimg[i,j] >= x) and (gmimg[i,j] >= y):
                nmsimg[i,j] = gmimg[i,j]
            else:
                nmsimg[i,j] = 0

    return nmsimg

#Double tresholding

def dbl_thresh(nmsimg,gaimg,t1,t2):

    nmsimgn = nmsimg.shape[0] - 1
    nmsimgm = nmsimg.shape[1] - 1
    dtimg = np.zeros_like(nmsimg)

    # Making pixels that go out of kernel as undefined(-1)

    for i in range(kn + kxn):
        dtimg[i, :] = -1
        dtimg[nmsimgn-i, :] = -1
    for i in range(km + kxm):
        dtimg[:, i] = -1
        dtimg[:, nmsimgm-i] = -1

    #Double tresholding based on t1 and t2

    for i in range(kn+kyn, nmsimgn - kn-kyn+1):
        for j in range(km+kym, nmsimgm - km-kym+1):
            if nmsimg[i,j] < t1:
                dtimg[i,j] = 0
            elif nmsimg[i,j] > t2:
                dtimg[i,j] = 255
            elif t1 <= nmsimg[i,j] <= t2:
                for x in (-1,0,1):
                    for y in (-1,0,1):
                        if not (x == 0 and y == 0):
                            if (nmsimg[i+x,i+y] > t2) and (abs(gaimg[i+x,i+y] - gaimg[i,j]) <= 45):
                                dtimg[i,j] = 255
                                break
                            else:
                                dtimg[i,j] = 0

    return dtimg

#Main function

if __name__ == "__main__":
    imgpath = input("Enter the path to the image:")
    if not(os.path.exists(imgpath)):
        print("Image path is incorrect")
    else:
        t1 = int(input("Enter lower treshold value:"))
        t2 = 2*t1
        img = cv2.imread(imgpath,0)

        gimg = guass_blur(img)                                          #Function call to guassian blur
        cv2.imwrite('gs_img.bmp',np.round(gimg))                        #Rounding off float values to int for display

        gximg,gyimg,gmimg,gaimg = sobel(gimg)                           #Function call for applying sobel operator
        cv2.imwrite('gx_img.bmp', abs(np.round(gximg/4)))               #Rounding off and normalising values for display
        cv2.imwrite('gy_img.bmp', abs(np.round(gyimg/4)))               #Rounding off and normalising values for display
        cv2.imwrite('gm_img.bmp', abs(np.round(gmimg/math.sqrt(32))))   #Rounding off and normalising values for display

        nmsimg = nonmaxima_supress(gmimg,gaimg)                         #Function call for non-maxima supression
        cv2.imwrite('nms_img.bmp', abs(np.round(nmsimg/math.sqrt(32)))) #Rounding off and normalising values for display

        dtimg = dbl_thresh(nmsimg,gaimg,t1,t2)                          #Function call for double tresholding
        cv2.imwrite('dt_img.bmp',dtimg)



