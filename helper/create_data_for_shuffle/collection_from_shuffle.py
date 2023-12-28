import os
import shutil


ipp = 'j:/data/'
opp = 'j:/data_easy/train/'


def collection_from_shuffle(ipp, opp, it='2.jpg'):
    ins = set([imn[:-5] for imn in os.listdir(ipp)])
    for im in ins:
        shutil.copyfile(ipp + im + it, opp + im.split("_")[2] + '/' + im + it)
    print('Done!')


def collection_from_shuffle_non(ipp, opp, it='2.jpg'):
    ins = set([imn[:-5] for imn in os.listdir(ipp)])
    for im in ins:
        if (im.split('_')[6] == '0'):
            shutil.copyfile(ipp + im + it, opp + im.split("_")[2] + '/' + im + it)
    print('Done!')


collection_from_shuffle_non(ipp, opp)

