import torch
import numpy as np
from NETWORK import Model
import torch.backends.cudnn as cudnn
from progress_bar import progress_bar
from math import log10
import matplotlib.pyplot as plt
from utils import compute_ssim
import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
class Trainer(object):

    def __init__(self, config, training_loader, testing_loader):
        super(Trainer, self).__init__()
        self.name = "_iter12_picsize128"
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self, path):
        self.model = Model()   
        self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3])
        if os.path.exists(path):
            self.model = torch.load(path)
            print("="*80+'\n'+"We will load the model which has been trained!!!!!\n"+'='*80+'\n')
        print("we have", torch.cuda.device_count(), "GPUs")
        print("Start training: model"+self.name)
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)
        self.criterion.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-04, eps=1e-08)

    def save_model(self):
        model_out_path = "./model/model"+self.name+".pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint save to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target).cuda()
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss/(batch_num+1)))
        print("  Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        return  train_loss

    def test(self):
        self.model.eval()
        avg_psnr = 0
        name = 0
        pae = 6
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                loss = self.criterion(prediction, target)
                prediction = np.around(prediction.cpu()*128*pae*0.7 + 128)
                target = np.around(target.cpu()*128*pae*0.7 + 128)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(65025 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
            prediction = prediction.cpu().numpy()
            prediction = np.reshape(prediction,(prediction.shape[2],prediction.shape[3]))
            cv2.imwrite('./result/test.png',prediction)
        print("  Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
        return avg_psnr / len(self.testing_loader)


    def run(self):
        a = []
        b = []
        c = []
        d = []
        best_psnr = 0
        model_path = ''
        self.build_model(model_path)
        for epoch in range(1, self.nEpochs +1):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            # test_psnr = 100
            test_psnr = self.test()
            a.append(epoch)
            b.append(test_psnr)
            c.append(loss)
            d.append(self.optimizer.param_groups[0]['lr'])

            self.scheduler.step(loss)
           
            if test_psnr > best_psnr:
                self.save_model()
                best_psnr = test_psnr
            
            fig1 = plt.figure()
            plt.plot(a,b,'b-',linewidth=2)
            plt.xlabel('epoch')
            plt.ylabel('set14_psnr')
            plt.title('set14_psnr, best is: %.4f'%best_psnr)
            plt.grid()
            plt.savefig('./result/set14_psnr'+self.name+'.png')
            plt.close(fig1)

            fig2 = plt.figure()
            plt.plot(a,c,'b-',linewidth=2)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.grid()
            plt.title('loss')
            plt.savefig('./result/loss'+self.name+'.png')
            plt.close(fig2)

            fig3 = plt.figure()
            plt.plot(a,d,'b-',linewidth=2)
            plt.xlabel('epoch')
            plt.ylabel('learning rate')
            plt.grid()
            plt.title('learning rate')
            plt.savefig('./result/lr'+self.name+'.png')
            plt.close(fig3)

