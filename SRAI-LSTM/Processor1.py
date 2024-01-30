'''
Author: Pu Zhang
Date: 2019/7/1
'''
from utils import *
from utils1 import *
import yaml

import time
import torch.nn as nn
import numpy
# import os, cv2
# import matplotlib.pyplot as plt


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
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

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
        test_error, test_final_error = self.test_epoch4(self.args.load_model)
        print('Set: {}, epoch: {:.5f}, test_error: {:.5f} test_final_error: {:.5f}'.format(self.args.test_set, self.args.load_model, test_error, test_final_error))

    def playtrain(self):
        print('Training begin')
        # find_result = []
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            train_loss = self.train_epoch(epoch)
            # val_error, val_final = self.test_epoch4(epoch)

            # test
            if epoch > self.args.start_test:
                test_error, test_final_error = self.test_epoch4(epoch)
                self.save_model(epoch)

            # log files
            self.log_file_curve.write(str(epoch) + ', ' + str(train_loss) + ', ' + ', '+str(test_error) + ', ' + str(test_final_error) + '\n')

            if epoch % 10 == 0:
                self.log_file_curve.close()
                self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            # console log
            print('----epoch {}, train_loss={:.5f}, test_error={:.3f}, test_final={:.3f}'
                  .format(epoch, train_loss, test_error, test_final_error))

    def smaller(self, A, Aepoch, B, Bepoch):
        if A < B:
            return A, Aepoch
        else:
            return B, Bepoch
    
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
                print(
                'train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}'.format(batch, self.dataloader.trainbatchnums,
                                                                                epoch, loss.item(), end - start))
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums

        return train_loss_epoch

    def val_epoch(self, epoch):
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

    def test_epoch(self, epoch):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        # final_error_list = []
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5
        start = time.time()
        for batch in range(self.dataloader.testbatchnums):
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward
            self.net.zero_grad()
            outputs_infer = forward(inputs_fw, iftest=True)

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt, error_nl, error_nl_cnt, _ = \
                L2forTest_nl(outputs_infer, batch_norm[1:, :, :2], self.args.obs_length, lossmask,
                             seq_list[1:], nl_thred=0)
            # print('batch:', batch, error/error_cnt, final_error/final_error_cnt)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            error_nl_epoch += error_nl
            error_nl_cnt_epoch += error_nl_cnt
            # fn = fn.tolist()

            # final_text = self.args.model_dir + 'final_error/' + str(batch) + '.npy'
            # np.save(final_text, fn)
        end = time.time()
        times = end-start

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, \
               error_nl_epoch / error_nl_cnt_epoch, error_cnt_epoch, times/final_error_cnt_epoch, times/self.dataloader.testbatchnums

    def test_epoch1(self, epoch):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5
        # batch = 70  # 3:70 2:16 4:229

        for batch in range(self.dataloader.testbatchnums):
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward
            self.net.zero_grad()
            outputs_infer = forward(inputs_fw, iftest=True)
            prediction = outputs_infer + shift_value[1:]

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            lossmask1 = torch.sum(lossmask, 0)
            start, end = 0, 0
            for i in range(len(batch_id)):
                if batch_id[i][0] == 0:
                    end = start + int(batch_pednum[i])
                    lossmask2 = lossmask1[start:end]
                    future = prediction[:, start:end, :] # 三维矩阵，
                    groundtruth = batch_abs[:, start:end, :]
                    # print(future - groundtruth[1:])
                    ind = torch.eq(lossmask2, 19)
                    start = end
                    future = future[:, ind, :].cpu().detach().numpy()
                    groundtruth = groundtruth[:, ind, :].cpu().detach().numpy()
                    if groundtruth.shape[1] >= 1:
                        gt_text = self.args.model_dir + 'batch/' + str(batch_id[i][0]) + '_' + str(batch_id[i][1]) + 'gt'
                        pt_text = self.args.model_dir + 'batch/' + str(batch_id[i][0]) + '_' + str(batch_id[i][1]) + 'pt'
                        np.save(gt_text, groundtruth)
                        np.save(pt_text, future)
                    # visualize(self.args.test_set, batch_id[i][1], groundtruth, future)

            fn, error, error_cnt, final_error, final_error_cnt, error_nl, error_nl_cnt, _ = \
                L2forTest_nl(outputs_infer, batch_norm[1:, :, :2], self.args.obs_length, lossmask,
                            seq_list[1:], nl_thred=0)

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            error_nl_epoch += error_nl
            error_nl_cnt_epoch += error_nl_cnt


        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, \
               error_nl_epoch / error_nl_cnt_epoch, error_cnt_epoch

    def test_epoch2(self, epoch):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5
        path = self.args.model_dir + 'visualization/'
        H = numpy.loadtxt(path + 'H.txt').astype(np.float32)
        H_t = torch.pinverse(torch.from_numpy(H))
        if self.args.train_model == 'salstm':
            attention_id = 0
        elif self.args.train_model == 'ralstm':
            attention_id = 1
        elif self.args.train_model == 'sralstm':
            attention_id = 2

        for batch in range(self.dataloader.testbatchnums):
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward
            self.net.zero_grad()
            outputs_infer = forward(inputs_fw, iftest=True)
            prediction = outputs_infer + shift_value[1:]

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt, error_nl, error_nl_cnt, _ = \
                L2forTest_nl(outputs_infer, batch_norm[1:, :, :2], self.args.obs_length, lossmask,
                             seq_list[1:], nl_thred=0)
            print('batch:', batch, error/error_cnt, final_error/final_error_cnt)
            start, end = 0, 0
            lossmask1 = torch.sum(lossmask, 0)
            for i in range(len(batch_id)):
                end = start + int(batch_pednum[i])
                lossmask2 = lossmask1[start:end]
                future = prediction[:, start:end, :]
                groundtruth = batch_abs[:, start:end, :]
                # print("lossmask2:", lossmask2)
                ind = torch.eq(lossmask2, 19)
                start = end
                future = future[:, ind, :].cpu().detach().numpy()
                groundtruth = groundtruth[:, ind, :].cpu().detach().numpy()
                if groundtruth.shape[1] > 1:
                    gt_text = path + 'save_data/gt_' + str(batch) + '_' + str(batch_id[i][0]) + '_' + str(batch_id[i][1])
                    pt_text = path + 'save_data/pt_' + str(batch) + '_' + str(batch_id[i][0]) + '_' + str(batch_id[i][1]) + str(attention_id)
                    np.save(gt_text, groundtruth)
                    np.save(pt_text, future)
                    # self.visualization(groundtruth, future, H_t, path, batch_id[i][1])

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            error_nl_epoch += error_nl
            error_nl_cnt_epoch += error_nl_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, \
               error_nl_epoch / error_nl_cnt_epoch, error_cnt_epoch

    def test_epoch3(self, epoch):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5
        path = self.args.save_dir + 'results/'
        path1 = self.args.save_dir + 'batch_id/'
        path2 = self.args.model_dir + 'attention/'

        if self.args.train_model == 'salstm':
            attention_id = 0
        elif self.args.train_model == 'ralstm':
            attention_id = 1
        elif self.args.train_model == 'sralstm':
            attention_id = 2
        else:
            attention_id = 3

        for batch in range(self.dataloader.testbatchnums):
            inputs, batch_id = self.dataloader.get_test_batch(batch)
            batch_id_text = path1 + 'batch_id_' + str(batch)
            # np.save(batch_id_text, batch_id)
            # print(batch_id)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_fw = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]
            forward = self.net.forward
            self.net.zero_grad()
            outputs_infer, Attention = forward(inputs_fw, iftest=True)
            prediction = outputs_infer + shift_value[1:]

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            error, error_cnt, final_error, final_error_cnt, error_nl, error_nl_cnt, _ = \
                L2forTest_nl(outputs_infer, batch_norm[1:, :, :2], self.args.obs_length, lossmask,
                             seq_list[1:], nl_thred=0)
            print('batch:', batch, error/error_cnt, final_error/final_error_cnt)
            start, end = 0, 0
            lossmask1 = torch.sum(lossmask, 0)

            '''
            for i in range(len(batch_id)):
                end = start + int(batch_pednum[i])
                lossmask2 = lossmask1[start:end]
                future = prediction[:, start:end, :]
                groundtruth = batch_abs[:, start:end, :]
                # print("lossmask2:", lossmask2)
                ind = torch.eq(lossmask2, 19)
                start = end
                future = future[:, ind, :].cpu().detach().numpy()
                groundtruth = groundtruth[:, ind, :].cpu().detach().numpy()
                if groundtruth.shape[1] > 1:
                    gt_text = path + 'gt_' + str(batch_id[i][0]) + '_' + str(batch_id[i][1])
                    pt_text = path + 'pt_' + str(batch_id[i][0]) + '_' + str(batch_id[i][1]) \
                              + '_' + str(attention_id)
                    np.save(gt_text, groundtruth)
                    np.save(pt_text, future)
                    # self.visualization(groundtruth, future, H_t, path, batch_id[i][1])
            '''
            for i in range(len(batch_id)):
                end = start + int(batch_pednum[i])
                lossmask2 = lossmask1[start:end]
                attention1 = Attention[:, start:end, :]
                attention2 = attention1[:, :, start:end]

                ind = torch.eq(lossmask2, 19)
                start = end
                attention3 = attention2[:, ind, :]
                attention4 = attention3[:, :, ind].cpu().detach().numpy()
                if attention4.shape[1] > 1:
                    attention_text = path2 + 'att_' + str(batch_id[i][0]) + '_' + str(batch_id[i][1])
                    np.save(attention_text, attention4)
                    # self.visualization(groundtruth, future, H_t, path, batch_id[i][1])

            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt
            error_nl_epoch += error_nl
            error_nl_cnt_epoch += error_nl_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, \
               error_nl_epoch / error_nl_cnt_epoch, error_cnt_epoch

    def test_epoch4(self, epoch):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch, error_nl_epoch = 0, 0, 0
        error_cnt_epoch, final_error_cnt_epoch, error_nl_cnt_epoch = 1e-5, 1e-5, 1e-5

        for batch in range(self.dataloader.testbatchnums):
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
                                                                        self.args.obs_length, lossmask, num_samples=self.args.sample_num)
            prediction = all_output + shift_value[1:]

            lossmask, num = getLossMask(outputs_infer, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            lossmask1 = torch.sum(lossmask, 0)
            start, end = 0, 0
            for i in range(len(batch_id)):
                if batch_id[i][0] == 0:
                    end = start + int(batch_pednum[i])
                    lossmask2 = lossmask1[start:end]
                    future = prediction[:, -12:, start:end, :]
                    groundtruth = batch_abs[:, start:end, :]
                    ind = torch.eq(lossmask2, 19)
                    start = end
                    future = future[:, :, ind, :].cpu().detach().numpy()
                    groundtruth = groundtruth[:, ind, :].cpu().detach().numpy()
                    if groundtruth.shape[1] >= 1:
                        gt_text = self.args.model_dir + 'batch/' + str(batch_id[i][0]) + '_' + str(batch_id[i][1]) + 'gt'
                        pt_text = self.args.model_dir + 'batch/' + str(batch_id[i][0]) + '_' + str(batch_id[i][1]) + 'pt'
                        error_ = np.linalg.norm(future - groundtruth[-12:], ord=2, axis=3).sum(axis=1)
                        min_index = np.argmin(error_, axis=0)
                        save_futures = []
                        for i in range(len(min_index)):
                            save_futures.append(future[:, :, i, :][min_index[i]])
                        save_futures = np.transpose(np.stack(save_futures), axes=(1, 0, 2))
                        np.save(gt_text, groundtruth)
                        np.save(pt_text, save_futures)
            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch

