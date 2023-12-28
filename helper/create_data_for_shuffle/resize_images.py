import os
from PIL import Image


ipp = 'j:/data1234/'
opp = 'j:/data/'


def resize_images():
    ims = os.listdir(ipp)
    for im in ims:
        img = Image.open(ipp + im)
        img = img.resize((320, 320), Image.LANCZOS)
        img.save(opp + im, 'jpeg')
    print('Done!')


resize_images()

