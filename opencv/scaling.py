from PIL import Image
import os
import PIL
import glob

image = Image.open('opencv/resized_img.jpg')
print(image.size)
#(750, 610)
resized_img = image.resize((416,416))
print(resized_img.size)
#resized_img.save('resized_img.jpg')
