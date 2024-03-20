# LOD-net
## 1.Data preparation
Downloading training and testing datasets and move them into ./dataset/, which can be found in this [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view).
## 2.pretrained model
You can download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV) and then put it in the ./pretrained_pth.
## 3.Training
```
git clone https://github.com/pku-lyk/LOD-net.git
cd LOD-net
bash train.sh
```
## 4.Testing
```
cd LOD-net
bash test.sh
```

## 5.Evaluating your trained model
```
cd LOD-net
python Eval.py
```
## 6.Our trained model
You can download the trained model [Google Drive](https://drive.google.com/file/d/1FHTR_YEnuU1UCTuUoVSoETrJj6Q_2jtO/view?usp=drive_link) and put it in the ./model_pth

## 7.Acknowledgement
Thanks these excellent works [Pranet](https://github.com/DengPingFan/PraNet),[Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) which have provided the basis for our framework
