import os
import shutil


def load_in_limits(
    ipp='F:/stack_torch_src/',
    opp='J:/ttt_data/data_final_limit_3/',
    p1=['train/', 'valid/'],
    p2=['2/', '3/', '4/', '5/', '6/'],
    # result _ _ _ _ _ _ 叶片数量 长度 频率 噪声 角度 速度 .jpg
    start=8,
    limits=[
        [2., 6.5],
        [1, 30],
        [0, 30],
        [0, 60],
        [0, 20]
    ]
):
    for x in p1:
        for y in p2:
            ims = os.listdir(ipp + x + y)
            for im in ims:
                il = im.split('_')[start: -1]
                flag = True
                for i in range(len(il)):
                    if float(il[i]) < limits[i][0] or float(il[i]) > limits[i][1]:
                        flag = False
                if flag:
                    shutil.copyfile(ipp + x + y + im, opp + x + y + im)
    for x in p1:
        for y in p2:
            ims = os.listdir(opp + x + y)
            # for im in ims:
            #     os.remove(opp + x + y + im)
            print(len(ims))
    print('Done!')


load_in_limits()
