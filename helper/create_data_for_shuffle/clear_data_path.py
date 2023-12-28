import os


data_path = 'j:/data_easy/'
p1 = ['train/', 'valid/']
p2 = ['2/', '3/', '4/', '5/', '6/']


def clear_data_path():
    for x in p1:
        for y in p2:
            ims = os.listdir(data_path + x + y)
            for im in ims:
                os.remove(data_path + x + y + im)
    print('Done!')


clear_data_path()
