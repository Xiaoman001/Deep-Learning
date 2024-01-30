'''
Train script
Author: Pu Zhang
Date: 2019/7/1
'''
import argparse
import ast
from Processor1 import *


def get_parser():

    parser = argparse.ArgumentParser(
        description='Social Relationship Attention LSTM')
    parser.add_argument(
        '--using_cuda', default=True)  # We did not test on cpu
    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set', default=1,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scrits.')
    parser.add_argument('--save_base_dir', default='./savedata/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='test', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='sralstmm', help='Your model name')
    parser.add_argument('--load_model', default=199,
        help="load model weights from this index before training or testing")
    parser.add_argument(
        '--pretrain_model', default='sralstmm',
        help='Your pretrained model name. Used in training second states refienemnt layer.')
    parser.add_argument(
        '--pretrain_load', default=0,
        help="load pretrained model from this index. Used in training second states refienemnt layer.")
    parser.add_argument(
        '--model', default='models.SRA_LSTM_M')
    ######################################

    parser.add_argument(
        '--dataset', default='eth5')
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')

    parser.add_argument(
        '--ifvalid', default=True, 
        help="=False, use all train set to train, "
             "=True, use train set to train and valid")
    parser.add_argument(
        '--val_fraction', default=0.0)
    parser.add_argument(
        '--sample_num', default=20
    )

    # Model parameters

    # LSTM
    parser.add_argument(
        '--input_size', default=2, type=int)
    parser.add_argument(
        '--output_size', default=2, type=int)
    parser.add_argument(
        '--input_embed_size', default=32, type=int)
    parser.add_argument(
        '--rnn_size', default=64, type=int)
    parser.add_argument(
        '--ifdropout', default=True, type=ast.literal_eval)
    parser.add_argument(
        '--dropratio', default=0.5, type=float)
    parser.add_argument(
        '--std_in', default=0.2, type=float)
    parser.add_argument(
        '--std_out', default=0.1, type=float)
    parser.add_argument(
        '--noise_dim', default=16, type=int)

    # Relation LSTM
    parser.add_argument(
        '--rela_input', default=2)
    parser.add_argument(
        '--rela_embed_size', default=32)
    parser.add_argument(
        '--rela_hidden_size', default=64)
    parser.add_argument(
        '--rela_dropratio', default=0.5)
    parser.add_argument(
        '--rela_ac', default='relu')

    # Social Interaction Module
    parser.add_argument(
        '--social_tensor_size', default=64)

    # Perprocess
    parser.add_argument(
        '--seq_length', default=16)
    parser.add_argument(
        '--obs_length', default=8)
    parser.add_argument(
        '--pred_length', default=12)
    parser.add_argument(
        '--batch_around_ped', default=128)
    parser.add_argument(
        '--batch_size', default=32)
    parser.add_argument(
        '--val_batch_size', default=32)
    parser.add_argument(
        '--test_batch_size', default=16)
    parser.add_argument(
        '--show_step', default=100)
    parser.add_argument(
        '--start_test', default=-1)
    parser.add_argument(
        '--num_epochs', default=300)
    parser.add_argument(
        '--ifshow_detail', default=True)
    parser.add_argument(
        '--ifdebug', default=False)
    parser.add_argument(
        '--ifsave_results', default=False)
    parser.add_argument(
        '--randomRotate', default=True,
        help="=True:random rotation of each trajectory fragment")
    parser.add_argument(
        '--neighbor_thred', default=2)
    parser.add_argument(
        '--learning_rate', default=0.0015)
    parser.add_argument(
        '--clip', default=1)
    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    p.save_dir = p.save_base_dir+str(p.test_set)+'/'
    p.model_dir = p.save_base_dir+str(p.test_set)+'/'+p.train_model+'/'
    p.config = p.model_dir+'/config_'+p.phase+'.yaml'

    # if not load_arg(p):
    save_arg(p)
    args = load_arg(p)
    args.seq_length = args.obs_length + args.pred_length
    torch.cuda.set_device(args.gpu)
    processor = Processor(args)
    if args.phase == 'test':
        processor.playtest()
    else:
        processor.playtrain()
