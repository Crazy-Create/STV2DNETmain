#  STV2DNET-main
![图4](https://github.com/user-attachments/assets/a2aa915d-ebbb-4038-95d6-52f9cab47b51)<br>
网络架构如图所示：<br>
![图1](https://github.com/user-attachments/assets/491cb991-273a-4537-b996-4b59d93ff5c8)<br>
论文链接请参考知网：https://kns.cnki.net/kcms2/article/abstract?v=oWJgMrFo8ufmNc9VHNrm3qsYqQXce_3r8b-zY1bANsKwhXRgSSLYanjedf86jlgcB-RjVIbS39XUTG3GkzgrBfS-1Rhd_szwWYAsn8Y64YrMLoq5b-NYAZigAdT_O6kvmPA2FSOePmZ1b9DWNFTQtM1Y_TgZzJGkwOsu2WOtdU0=&uniplatform=NZKPT<br>
这个项目工程的代码是基于 pytorch，python3.8测试实现的。 它的预训练权重文件在以下百度网盘的链接中：https://pan.baidu.com/s/1fbUMHkPjKCzt0bkyUn5ACg?pwd=z4us password: z4us <br>

```
需要安装的头文件,没有版本限制安装最新版即可
pip install pandas
pip install matplotlib
pip install tqdm
pip install argparse
```

首先测试部分：

## test.py文件测试<br>
测试部分文件夹构建请参考以下：
```
┬─ save_models
│   ├─ indoor
│   │   ├─ dehazeformer-s.pth
└─ data
    ├─ RESIDE-IN
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       └─ ...
```
若模型进入test模式，输出**空套件​**​等信息。
请在pycharm中点击“文件”然后打开“设置”选择“工具”，在默认测试运行程序中选择“Unittest”然后点击“应用”然后“确定”。
最终输入效果应该如图所示。<br>
<div align="center">
<img width="488" height="199" alt="image" src="https://github.com/user-attachments/assets/f779e024-4aff-47a7-9077-82808aafa9f2" />
</div>
<br><br><br><br>

## test_out.py
此文件是输入单张带雾图片输出去雾后的图片。<br><br><br><br>

训练阶段，训练阶段运行
## train.py
在代码中

```
--model   dehazeformer-s     指定模型文件
--save_dir  ./saved_models/  指定训练完成后的保存的权重文件的地址
--data_dir  ./data/          数据集存放地址
--dataset   RESIDE-6K        指定数据集
--gpu       '0,1,2,3,4,5'    指定gpu数量
```
