'''
Author: Pu Zhang
Date: 2019/7/1
'''
from utils import *
import logging
import time
import torch.nn as nn
import yaml
from tqdm import tqdm


class Processor():
    def __init__(self, args):
        self.args = args

        Dataloader = DataLoader_bytrajec2

        self.dataloader = Dataloader(args)
        model = import_class(args.model)
        self.net = model(args)
        self.set_optimazier()
        self.load_model()

        # self.load_weights_from_srlstm()
        # Uncomment to train the second SR layer

        if self.args.using_cuda:
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()
        print(self.net)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):
        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
        torch.save({
            'epoch': epoch, 
            'state_dict': self.net.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):
        if self.args.load_model > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)  # , map_location={'cuda:1': 'cuda:2'})
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def load_model_epoch(self, epoch):
        if epoch > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)  # , map_location={'cuda:2': 'cuda:0'})
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimazier(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss(reduce=False)

    def playtest(self):
        print('Testing begin')
        test_error, test_final_error, _,  _ = self.test_epoch(self.args.load_model)
        print('Set: {}, epoch: {:.5f}, test_error: {:.5f} test_final_error: {:.5f}'.format(self.args.test_set, self.args.load_model, test_error, test_final_error))

    def playtrain(self):
        train_log = 'train.log'
        set_logger(os.path.join(self.args.model_dir, train_log))

        logging.info('Training begin')
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch)

            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error = self.test_epoch()
                self.best_ade = test_error if test_final_error < self.best_fde else self.best_ade
                self.best_epoch = epoch if test_final_error < self.best_fde else self.best_epoch
                self.best_fde = test_final_error if test_final_error < self.best_fde else self.best_fde
                self.save_model(epoch)

            self.log_file_curve.write(
                str(epoch) + ',' + str(train_loss) + ',' + str(test_error) + ',' + str(test_final_error) + ',' + str(
                    self.args.learning_rate) + '\n')

            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            if epoch >= self.args.start_test:
                logging.info(
                    '----epoch {}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                        .format(epoch, train_loss, test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                logging.info('----epoch {}, train_loss={:.5f}'.format(epoch, train_loss))

    def train_epoch(self, epoch):
        self.dataloader.reset_batch_pointer(set='train', valid=False)

        loss_epoch = 0

        for batch in range(self.dataloader.trainbatchnums):
            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]

            self.net.zero_grad()

            outputs = self.net.forward(inputs_fw, iftest=False)

            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]), dim=2)

            loss += torch.sum(loss_o*lossmask)/num
            loss_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end = time.time()
            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                logging.info(
                    'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                               self.dataloader.trainbatchnums,
                                                                                               epoch, loss.item(),
                                                                                               end - start))

        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums

        return train_loss_epoch

    def val_epoch(self):
        if self.dataloader.val_fraction == 0:
            return 0, 0, 0
        self.dataloader.reset_batch_pointer(set='train', valid=True)
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5

        for batch in range(self.dataloader.valbatchnums):

            inputs, batch_id = self.dataloader.get_val_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward

            outputs_infer = forward(inputs_fw, iftest=True)
            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)

            error, error_cnt, final_error, final_error_cnt, _ = L2forTest(outputs_infer, batch_norm[1:, :, :2],
                                                                          self.args.obs_length, lossmask)
            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        val_error = error_epoch / error_cnt_epoch
        final_error = final_error_epoch / final_error_cnt_epoch

        return val_error, final_error

    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5

        for batch in tqdm(range(self.dataloader.testbatchnums)):
            #if batch % 100 == 0:
                #print('testing batch', batch, self.dataloader.testbatchnums)
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward
            all_output = []
            for i in range(self.args.sample_num):
                outputs_infer = forward(inputs_fw, iftest=True)
                all_output.append(outputs_infer)
            self.net.zero_grad()

            all_output = torch.stack(all_output)

            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_norm[1:, :, :2],
                                                                        self.args.obs_length, lossmask)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

    def cal_ade_fde(self, pred_traj_gt, pred_traj_fake):
        ade = displacement_error(pred_traj_fake, pred_traj_gt)
        fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
        return ade, fde

