import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('a0.jpg')
img = np.transpose(img,[2,0,1]) # KPU requires NCHW format, 
								# while Tensorflow requires NHWC.
with open('image.c','w') as f:
    print('const unsigned char gImage_image[]={', file=f)
    print(', '.join([str(i) for i in img.flatten()]), file=f)
    print('};', file=f)