import numpy as np
import cv2
import sys
from PIL import Image
import gc
import h5py
Crop_Scale = 512
Nonzero_Thresh = 0.3
Crop_Stride = int(Crop_Scale/2)
def cal_nonzero_ratio(im):
	return np.count_nonzero(im)/(Crop_Scale*Crop_Scale*3)
def one_to_three(ar):
	res = np.zeros((ar.shape[0],ar.shape[1],3))
	res[ar==1,0]=255
	res[ar==2,1]=255
	res[ar==3,2]=255
	return res
Image.MAX_IMAGE_PIXELS = None
img = Image.open('image_2.png')
img = np.asarray(img)
print(img.shape)
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

print(img.shape)
label = Image.open('image_2_label.png')
label = np.asarray(label)
print(label.shape)

img_width = img.shape[0]
img_height = img.shape[1]
train_id=1
for x in range(int(img_width/Crop_Stride)-1):
	for y in range(int(img_height/Crop_Stride)-1):
		tmp = img[x*Crop_Stride:x*Crop_Stride+Crop_Scale,y*Crop_Stride:y*Crop_Stride+Crop_Scale,:]
		if cal_nonzero_ratio(tmp)< Nonzero_Thresh:
			continue
		cv2.imwrite('crop2/'+'2_'+str(train_id)+'.png',tmp)
		tmp_label = label[x*Crop_Stride:x*Crop_Stride+Crop_Scale,y*Crop_Stride:y*Crop_Stride+Crop_Scale]
		tmp_label = one_to_three(tmp_label)
		tmp_label = tmp_label.astype(np.uint8)

		np.save('crop2/'+'2_'+str(train_id)+'.npy',tmp_label)

		train_id += 1
		print(train_id)

