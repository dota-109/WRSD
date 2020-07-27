import argparse
import torchvision
from torch.utils.data import DataLoader
from Trainer import Trainer
from DataSet import DataSetFromFolder

parser = argparse.ArgumentParser(description='WRSD')

parser.add_argument('--batchSize', type=int, 
	default=32, help='training batch size')

parser.add_argument('--testBatchSize', type=int, 
	default=1, help='testing batch size')

parser.add_argument('--nEpochs', type=int, 
	default=150, help='number of epochs to train for')

parser.add_argument('--lr', type=float, 
	default=0.0005, help='Learning Rate. Default=0.0001')

parser.add_argument('--seed', type=int, 
	default=123, help='random seed to use. Default=123')

parser.add_argument('--imageDir', type=str, 
	default='./DataSet/DIV2K_clip_128/PNG/')

parser.add_argument('--targetDir', type=str, 
	default='./DataSet/DIV2K_clip_128/PNG/gray/')

parser.add_argument('--testDir', type=str, 
	default='./DataSet/Set14/')

parser.add_argument('--testLabelDir', type=str, 
	default='./DataSet/Set14/gray/')


args = parser.parse_args()



def main():
	print('===> Loading datasets')
	train_set = DataSetFromFolder(image_dir=args.imageDir, target_dir=args.targetDir, if_test=False,pae=0)
	test_set = DataSetFromFolder(image_dir=args.testDir, target_dir=args.testLabelDir, if_test=True,pae=6)
	training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True, num_workers=8, pin_memory=True)
	testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False, num_workers=8, pin_memory=True)

	model = Trainer(args, training_data_loader, testing_data_loader)
	model.run()

if __name__ == '__main__':
	main()
