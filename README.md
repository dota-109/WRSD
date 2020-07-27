# WRSD
**"CALIC Soft Decoding Using Hard Constrained Wide-activated Recurrent Residual Network."**

## Dependencies
* Python 3.6
* PyTorch >= 1.1.0
* numpy
* matplotlib
* cv2 >= 3.x.x 
* pillow

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/dota-109/WRSD
cd WRSD
```

## Datasets
You can download test datasets from:
### Baidu Pan
```
https://pan.baidu.com/s/1TIGEeh11Ok9P3ziE5Sm7rw
code: 9bc8
```
### Google Drive
```
https://drive.google.com/file/d/1aGDeMYPa91PCgZqE4D6fK3pOkHjPjdar/view?usp=sharing
```

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
                    [--testBatchSize TEST_BATCHSIZE] [--seed SEED]
                    [--imageDir TRAINING_SET_DATA_DIR_PATH]
                    [--targetDir TRAINING_SET_LABEL_DIR_PATH]
                    [--testDir TESTING_SET_DATA_DIR_PATH]
                    [--testLabelDir TESTING_SET_LABEL_DIR_PATH]
               
optional arguments:
  -h, --help            Show this help message and exit
  --batchSize           Training batch size. Default=32
  --nEpochs             Number of epochs to train for. Default=150
  --lr                  Learning rate. Default=0.0005
  --testBatchSize       Testing batch size. Default=1
  --imageDir            Training set data dir path
  --targetDir           Training set label dir path
  --testDir             Testing set data dir path
  --testLabelDir        Testing set label dir path
```
An example of training usage is shown as follows:
```
python main.py --batchSize 16 --nEpochs 100
```
## Result
From left to right are calic and wrsd
<p>
  <img src='result/woman_calic.png' height='400' width='400'/>
  <img src='result/woman_wrsd.png' height='400' width='400'/>
</p>
