import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import misc
import math
from numpy import r_
from math import cos
from PIL import Image
from matplotlib.image import imread

class DCTController:

    def get_matrix_from_image(test, path, mode="RGB"):
        x = Image.open(r"C:\Users\Dauma\Desktop\test.jpg").convert(mode)
        data = np.asarray(x)
        return(data)

    def cosp(test,i,j,n):
        output = 0
        output = cos(((2*i)+1)*j*math.pi/(2*n))
        return output

    def convolveDCT(test,f,n,u,v,a,b): # This convolve function compute DCT for nxn
        sumd = 0                             
        for x in r_[0:n]:
            for y in r_[0:n]:
                u = u%n
                v = v%n
                sumd += f[x+a,y+b]*DCTController().cosp(x,u,n)*DCTController().cosp(y,v,n)
   
        if u == 0: sumd *= 1/math.sqrt(2) 
        else: sumd *= 1
        if v == 0: sumd *= 1/math.sqrt(2)
        else: sumd *= 1
        sumd *= 1/math.sqrt(2*n)

        return sumd

    def convolveIDCT(test,dctmatrix,n,x,y,a,b): # This convolve function compute DCT for nxn
        for u in r_[0:n]:
            for v in r_[0:n]:
                val1 = 1
                val2 = 1
                x = x%n
                y = y%n
                if u == 0: val1 = 1/math.sqrt(2)
                if v == 0: val2 = 1/math.sqrt(2)
                sumd += dctmatrix[u+a,v+b]*val1*val2*DCTController().cosp(x,u,n)*DCTController().cosp(y,v,n)   
        sumd *= 2/n
        return sumd

    def do_compress(test, f, quality):
        img = Image.open(f)
        gray_img = img.convert("L")
        f = np.asarray(gray_img)
        plt.imshow(f, cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()
        print('image matrix size: ', f.shape )
        n = 8  # This will be the window in which we perform our DCT
        sumd = 0 # INIT value

        pos = 175
        size = 256
        f = f[pos:pos+size,pos:pos+size]

        # Create some blank matrices to store our data

        dctmatrix = np.zeros(np.shape(f)) # Create a DCT matrix in which to plug our values
        f = f.astype(np.int16) # Convert so we can subtract 128 from each pixel needed otherwise there is an out of bounds errors
        f = f-128
        f2 = np.zeros(np.shape(f))
        # First we need to take into account our multiple nxn windows that jump across the image
        for a in r_[0:np.shape(f)[0]:n]:
            for b in r_[0:np.shape(f)[1]:n]:
                # Below, compute the DCT for a given uxv location in the DCT Matrix
                for u in r_[a:a+n]:
                    for v in r_[b:b+n]:
                        print DCTController().convolveDCT(f,n,u,v,a,b)
                        dctmatrix[u,v] = DCTController().convolveDCT(f,n,u,v,a,b)
        np.around(dctmatrix)

        plt.figure()
        plt.imshow(dctmatrix,cmap='gray',vmax = np.max(dctmatrix)*0.01,vmin = 0)
        plt.title("8x8 DCTs of the image")
        plt.show()

# Basic quant table
        Quant = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        # Finding adjustement for how much needs to be reduced depending on the provided required quality
        if quality <= 50:
            scaling_factor = 50 / quality
        else:
            scaling_factor = 2 - (quality / 50)
        for a in r_[0:np.shape(f)[0]:n]:
            for b in r_[0:np.shape(f)[1]:n]:
                dctmatrix[a:a+n,b:b+n] = dctmatrix[a:a+n,b:b+n]/Quant*scaling_factor

        # First we need to take into account our multiple nxn windows that jump across the image
        for a in r_[0:np.shape(dctmatrix)[0]:n]:
            for b in r_[0:np.shape(dctmatrix)[1]:n]:
                # Below, compute the IDCT for a given x,y location in the Image Matrix
                for x in r_[a:a+n]:
                    for y in r_[b:b+n]:
                        f2[x,y] = DCTController().convolveIDCT(dctmatrix,n,x,y,a,b)

        print f2
        f2 = f2 + 128 # Scale our values back to 0-255 so we can see it!

        plt.figure()
        plt.imshow(f2, cmap=plt.cm.gray)
        plt.title("8x8 DCTs of the image3")
        plt.show()

        return(f2)


    def save_image(test, matrix, name):
        img = Image.fromarray(np.asarray(
            np.clip(matrix, 0, 255), dtype="uint8"), "L")
        img.save(str(name), 'png')
        return(None)
