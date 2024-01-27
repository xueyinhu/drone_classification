import os
import shutil


inp = 'j:/data_pre/data/'
oup = 'j:/dn_data/'


# '_0_*_0_2.jpg'
# result_30_2_5.1_1_0_6_0_2.jpg
# 恒定 [0, 6, 12, 18, 24, 30]
def collection_from_shuffle():
    ims = os.listdir(inp)
    ips = set([im[:(-11 - len(im.split('_')[6]))] for im in ims])
    for idx, imp in enumerate(ips):
        if int(imp.split('_')[1]) <= 40:
            omp = oup + 'train/'
        else:
            omp = oup + 'valid/'
        for nos in ['0', '6', '12', '18', '24', '30']:
            shutil.copyfile(inp + imp + '_0_' + nos + '_0_2.jpg', omp + str(idx)+ '_' + nos + '.jpg')
    print("Done!")






