import os


def remove_checkpoints(p='J:/c/checkpoints/'):
    fns = os.listdir(p)
    for fn in fns:
        os.remove(p + fn)
    print('Done!')


remove_checkpoints()
