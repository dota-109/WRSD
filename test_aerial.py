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
from torch.utils.data import DataLoader

from DataSet_aerial import DataSetFromFolder

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

load_model_name = './model/model_iter12_picsize128.pth'
save_file_name = '_Aerial_dataset_' + str(load_model_name[:-4]) + '.txt'

def test(path, testing_loader, pae):
    model = torch.load(path)
    
    model.eval()
    mse = torch.nn.MSELoss()
    avg_ssim = 0
    avg_psnr = 0
    max_bound = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(testing_loader):
            data, target = data.cuda(), target.cuda()
            prediction = model(data)
            prediction = (prediction.cpu()*128*pae*0.7 + 128)
            target = (target.cpu() * 128 * pae * 0.7 + 128)
            
            mse_value = mse(prediction, target)
            psnr = 10 * log10(65025 / mse_value.item())
            avg_psnr += psnr
            progress_bar(batch_num, len(testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
            abs_value = np.abs(prediction-target)
            abs_value = abs_value.numpy()
            if abs_value.max() >max_bound:

                max_bound = abs_value.max()
            prediction = prediction.numpy()

            prediction = np.reshape(prediction,(prediction.shape[2],prediction.shape[3]))
            target = np.reshape(target,(target.shape[2],target.shape[3]))
            ssim = compute_ssim(prediction, target)
            avg_ssim += ssim
            # break

    print("  Average PSNR: {:.4f} dB".format(avg_psnr / len(testing_loader)))
    print("  Average SSIM: {:.4f} dB".format(avg_ssim / len(testing_loader)))
    print("bound: " + str(max_bound))
    return avg_psnr / len(testing_loader), avg_ssim / len(testing_loader), max_bound


def main():
    
    for pae in range(6,11,2):
    
        fp = open('./Aerial_result/psnr_'+str(pae)+save_file_name,'w')
        
        print('===> Loading datasets')
        for i in range(180):
            print('='*5+'This is the '+str(i+1)+'th image' )
            testDir = '/home/ubuntu/ADAXI/Data_Set/Aerial/PNG_clip/' + str(pae) + '/' + str(i) + '/'
            testLabelDir = '/home/ubuntu/ADAXI/Data_Set/Aerial/PNG_clip/gray/' + str(i) + '/'

            test_set = DataSetFromFolder(image_dir=testDir, target_dir=testLabelDir, if_test=True,pae=pae)
            testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

            psnr_s, ssim_s, bound_s = test(load_model_name, testing_data_loader, pae)

            fp.writelines(str(i) + ' ' + 'PSNR: ' + str(psnr_s) + ' SSIM: ' + str(ssim_s) + ' bound: ' + str(bound_s) + '\n')
            # break
        fp.close()

if __name__ == '__main__':
    main()
