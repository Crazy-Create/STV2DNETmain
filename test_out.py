import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from models import dehazeformer_s  # 根据实际模型文件导入
from utils import chw_to_hwc, write_img

# ============== 硬编码路径配置区 ==============
img_path = r"C:\Users\ASUS\Desktop\文献\一种基于光流动态特征强化注意力机制铁路作业场景去雾方法\实验图片\各网络比较图\hazy2.jpg"  # 输入雾图路径
output_dir = r"D:\dehazed_results"  # 输出目录
weights_path = r"dehazeformer-s.pth"  # 模型权重路径
model_type = "dehazeformer-s"  # 模型类型名称
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================

def load_model(weights, model_name):
    """加载去雾模型"""
    # 初始化模型结构
    model = eval(model_name.replace('-', '_'))()

    # 加载权重（适配单卡/多卡训练保存格式）
    state_dict = torch.load(weights)['state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model.cuda().eval()


def dehaze_single_image():
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = load_model(weights_path, model_type)

    # 读取并预处理图像
    img = Image.open(img_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).cuda() * 2 - 1  # [0,1] -> [-1,1]

    # 执行去雾
    with torch.no_grad():
        output = model(img_tensor).clamp_(-1, 1)

    # 后处理并保存
    output_np = chw_to_hwc((output * 0.5 + 0.5).squeeze(0).cpu().numpy())
    output_name = f"dehazed_{os.path.basename(img_path)}"
    write_img(os.path.join(output_dir, output_name), output_np)
    print(f"去雾完成！结果保存至：{os.path.join(output_dir, output_name)}")


if __name__ == '__main__':
    dehaze_single_image()