import os
import shutil


ipp = 'j:/total/train/'
opp = 'j:/total/valid/'


def split_valid_data(ipp, opp):
    ids = os.listdir(ipp)
    for id in ids:
        ims = os.listdir(ipp + id)
        for im in ims:
            if im.split('_')[1] in ['41', '42']:
                shutil.move(ipp + id + '/' + im, opp + id + '/' + im)
    print('Done!')


split_valid_data(ipp, opp)

