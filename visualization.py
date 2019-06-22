import numpy as np
import cv2

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
img = Image.open('image_1.png')
img = np.asarray(img)
print(img.shape)
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

print(img.shape)
label = Image.open('image_1_label.png')
label = np.asarray(label)
print(label.shape)

img[label==1,0]=255
img[label==2,1]=255
img[label==3,2]=255
cv2.imwrite('1.png',img)

