import argparse

def get_config():
    psr = argparse.ArgumentParser()

    psr.add_argument('--train_images_num', default=1485),
    psr.add_argument('--valid_images_num', default=270)

    psr.add_argument('--net_name', default='MyNet')
    psr.add_argument('--epoch_count', default=80000)
    psr.add_argument('--batch_size', default=32)
    psr.add_argument('--optimizer', default='adam')
    psr.add_argument('--loss_fn', default='CategoricalCrossentropy')
    psr.add_argument('--metrics', default=['accuracy'])
    psr.add_argument('--save_path', default="J:/c/checkpoints/")
    psr.add_argument('--stop_patience', default=5)
    psr.add_argument('--rdlr_rate', default=.1)
    psr.add_argument('--rdlr_patience', default=3)

    psr.add_argument('--model_range', default=[2, 6])
    psr.add_argument('--input_size', default=(320, 320, 3))

    psr.add_argument('--data_path', default='J:/total/')
    psr.add_argument('--data_image_size', default=(320, 320))
    psr.add_argument('--data_idg_train_cfg', default='train/')
    psr.add_argument('--data_idg_valid_cfg', default='valid/')

    psr.add_argument('--gpu', default='/gpu:0')
    psr.add_argument('--log_level', default='3')

    return psr.parse_args()


