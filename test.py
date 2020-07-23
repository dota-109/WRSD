import torch
import numpy as np
from NETWORK import Model
import torch.backends.cudnn as cudnn
from progress_bar import progress_bar
from math import log10
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils import KL, compute_ssim
import cv2
import os
import time
from torch.utils.data import DataLoader

from DataSet import DataSetFromFolder

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
load_model_name = './model/model_iter12_picsize128.pth'

def test(path, testing_loader, pae):
    i = 1
    model = torch.load(path, map_location='cuda:0')
    model.eval()
    mse = torch.nn.MSELoss()
    avg_ssim = 0
    avg_psnr = 0
    max_bound = 0
    avg_time_consume = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(testing_loader):
            data, target = data.cuda(), target.cuda()
            time_start = time.time()
            prediction = model(data)
            torch.cuda.synchronize()
            time_end = time.time()
            avg_time_consume += (time_end-time_start)
            prediction = (prediction.cpu()*128*pae*0.7 + 128)
            target = (target.cpu() * 128 * pae * 0.7 + 128)
            mse_value = mse(prediction, target)
            psnr = 10 * log10(65025 / mse_value.item())
            avg_psnr += psnr
            abs_value = np.abs(prediction-target)
            abs_value = abs_value.numpy()
            if abs_value.max() >max_bound:

	            max_bound = abs_value.max()
            prediction = prediction.numpy()
            
            prediction = np.reshape(prediction,(prediction.shape[2],prediction.shape[3]))
            target = np.reshape(target,(target.shape[2],target.shape[3]))
            ssim = compute_ssim(prediction, target)
            avg_ssim += ssim
            # print(str(i) + ' psnr:' + str(psnr) + ' ssim: ' + str(ssim))
            # cv2.imwrite('./test_result/epoch5/test{}.png'.format(i),prediction)
            i += 1
    print("  The pae is: {}".format(pae))
    print("  Average PSNR: {:.2f} dB".format(avg_psnr / len(testing_loader)))
    print("  Average SSIM: {:.4f} dB".format(avg_ssim / len(testing_loader)))
    print("  bound: {:.3f}".format(max_bound))
    print("  Average time consumption: {:.3f} s".format(avg_time_consume/len(testing_loader)))

def main():
# Set5 Set14 BSD100 Kodim
    dataset_names = ['Set5', 'Set14', 'BSD100', 'Kodim']
    dataset_names = ['LIVE1']
    # dataset_names = ['Kodim']
    for dataset_name in dataset_names:
        print('===> Loading datasets: '+dataset_name)
        for i in range(6,11,2):
            pae = i
            print('PAE is: '+str(pae))
            testDir = '/home/ubuntu/ADAXI/Data_Set/'+dataset_name+'/'
            testLabelDir = '/home/ubuntu/ADAXI/Data_Set/'+dataset_name+'/gray/'          

            test_set = DataSetFromFolder(image_dir=testDir, target_dir=testLabelDir, if_test=True, pae=pae)

            testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

            test(load_model_name, testing_data_loader, pae)

if __name__ == '__main__':
   
    main()
   
   
