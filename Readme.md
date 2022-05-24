# FS2K识别

## 环境配置

* python == 3.7.11
* pytorch == 1.10.0
* tqdm == 4.62.3
* torchvision == 0.11.1
FS2K数据集可以在[这里](https://pan.baidu.com/s/1ZdgoONyseQDaMbk5jYxmuQ?pwd=h6y8)获取，下载后解压到项目根目录即可

## 训练与测试

`python main.py` 来进行训练和测试，如果想要单独训练或者测试，可以注释代码来实现。

本代码提供不同模型与版本，版本v1代表不对style做识别，v2代表不对style,hair_color做识别，请在main.py中自行修改，模型类别如下

|  模型名称   |                     说明                      |
| :---------: | :-------------------------------------------: |
|  ConvNeXt   |  使用模型为convnext_base_22k_224预训练并微调  |
|  ConvNeXt   |  使用模型为convnext_base_22k_224预训练并微调  |
| ConvNeXtv2  | 使用模型为convnext_xlarge_22k_224预训练并微调 |
| ConvNeXtv2  | 使用模型为convnext_xlarge_22k_224预训练并微调 |
|  ResNet50   |       使用模型为ResNet50且预训练并微调        |
| ResNet50v2  |       使用模型为ResNet50但从头开始训练        |
|  ResNet152  |       使用模型为ResNet152且预训练并微调       |
| ResNet152v2 |       使用模型为ResNet152但从头开始训练       |



## 结果

请在result文件夹中查看结果，result文件夹内容详解如下：

* ./loss：每训练一次会讲训练过程中的损失保存
* ./model：每训练一次会保存对应的模型
* ./result：每次测试的预测结果
