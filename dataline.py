import pandas as pd
import matplotlib.pyplot as plt

# 加载训练日志文件
log_file = 'saved_models/indoor/training_log.xlsx'  # 请替换为实际的路径
df = pd.read_excel(log_file)

# 提取数据
epochs = df['epoch']
train_loss = df['train_loss']
valid_psnr = df['valid_psnr']

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', color='blue', linestyle='-', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.legend()
plt.savefig('train_loss_curve.png')  # 保存为图片
plt.show()

# 绘制验证 PSNR 曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, valid_psnr, label='Validation PSNR', color='red', linestyle='-', marker='o')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('Validation PSNR per Epoch')
plt.grid(True)
plt.legend()
plt.savefig('valid_psnr_curve.png')  # 保存为图片
plt.show()
