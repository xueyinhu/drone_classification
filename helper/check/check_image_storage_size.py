import os
import numpy as np


p = "F:/stack_torch_src/train/6/" 
sizes_train = [
    55825.10060091905,
    55967.911055871205,
    56002.19402159536,
    56059.38693454826,
    56043.32816653004
]


def check_image_storage_size(path):
    ips = os.listdir(path)
    r = []
    [r.append(os.path.getsize(path + ip)) for ip in ips]
    return r


# r = check_image_storage_size(p)
# print(np.average(np.array(r, np.int64))/ 3.)
# print(np.average(np.array(sizes_train, np.float32)))
# 55980 byte
print(55980 / 1024.)
# 54.7 kb
