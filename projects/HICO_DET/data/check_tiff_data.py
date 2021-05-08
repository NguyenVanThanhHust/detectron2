import os
import os.path as osp
import cv2
from PIL import Image

img_folder = "../../data/HICO_DET/images/train2015"
list_img = next(os.walk(img_folder))[2]
print(len(list_img))
for img_name in list_img:
    imgPath = osp.join(img_folder, img_name)
    try:
        img = Image.open(imgPath)
        print(imgPath)
        exif_data = img._getexif()
    except ValueError as err:
        print(err)
        print("Error on image: ", img)
