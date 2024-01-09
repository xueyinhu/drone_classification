import argparse

def get_config():
    psr = argparse.ArgumentParser()

    psr.add_argument('--train_images_num', default=68894),
    psr.add_argument('--valid_images_num', default=7000)

    psr.add_argument('--net_name', default='MyNet')
    psr.add_argument('--epoch_count', default=80000)
    psr.add_argument('--batch_size', default=16)
    psr.add_argument('--save_steps', default=500)
    psr.add_argument('--optimizer', default='adam')
    psr.add_argument('--loss_fn', default='CategoricalCrossentropy')
    psr.add_argument('--metrics', default=['accuracy'])
    psr.add_argument('--save_path', default="J:/c/checkpoints/")
    psr.add_argument('--stop_patience', default=5)
    psr.add_argument('--rdlr_rate', default=.1)
    psr.add_argument('--rdlr_patience', default=3)

    psr.add_argument('--init_lr', default=1e-3)
    psr.add_argument('--lr_steps', default=[10000, 20000, 30000, 40000, 50000, 60000, 70000])
    psr.add_argument('--lr_rate', default=.5)

    psr.add_argument('--model_range', default=[2, 6])
    psr.add_argument('--input_size', default=(960, 320, 3))

    psr.add_argument('--data_path', default='F:/stack_torch_src/')
    psr.add_argument('--data_image_size', default=(960, 320))
    psr.add_argument('--data_idg_train_cfg', default='train/')
    psr.add_argument('--data_idg_valid_cfg', default='valid/')

    psr.add_argument('--gpu', default='/gpu:0')
    psr.add_argument('--log_level', default='3')

    return psr.parse_args()


