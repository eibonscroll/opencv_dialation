import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

im = np.zeros((10,10),dtype='uint8')
print(im);
plt.imshow(im)

im[0,1] = 1
im[-1,0]= 1
im[-2,-1]=1
im[2,2] = 1
im[5:8,5:8] = 1

print(im)
plt.imshow(im)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
print(element)

ksize = element.shape[0]

height,width = im.shape[:2]

dilatedEllipseKernel = cv2.dilate(im, element)
print(dilatedEllipseKernel)
plt.imshow(dilatedEllipseKernel)

border = ksize // 2
paddedIm = np.zeros((height + border * 2, width + border * 2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
paddedDilatedIm = paddedIm.copy()

# Create a VideoWriter object
# Use frame size as 50x50
###
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('dilated_video.avi', fourcc, 10.0, (50, 50))

###

for h_i in range(border, height + border):
    for w_i in range(border, width + border):
        # Apply dilation to the padded image
        region = paddedIm[h_i - border:h_i + border + 1, w_i - border:w_i + border + 1]
        paddedDilatedIm[h_i, w_i] = np.max(region * element)

        # Resize output to 50x50 before writing it to the video
        resizedFrame = cv2.resize(paddedDilatedIm, (50, 50))

        # Convert resizedFrame to BGR before writing
        resizedFrameBGR = cv2.cvtColor(resizedFrame.astype('uint8'), cv2.COLOR_GRAY2BGR)
        out.write(resizedFrameBGR)

# Release the VideoWriter object
###
out.release()
###
# Crop the final dilated image to the original image size
croppedDilatedIm = paddedDilatedIm[border:-border, border:-border]

plt.imshow(croppedDilatedIm, cmap='gray')
plt.title('Cropped Final Dilated Image')
plt.show()