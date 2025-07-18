import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img

# 这个函数对输入的图像进行随机裁剪、水平翻转和旋转等操作
def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay: # 决定边缘裁剪概率的参数
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip: # 如果为 True，则只进行水平翻转
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs

# 对图像进行中心裁剪，保证输出的图像大小是指定的 size
def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs

# 用于加载成对图像数据,成对图像分别放在GT，Haze文件中
# data_dir：数据集的根目录路径。
# sub_dir：子目录名，通常用于指定不同的数据集部分（例如 train、valid、test）。
# mode：设置加载模式，决定是训练模式 ('train')、验证模式 ('valid') 还是测试模式 ('test')。
# size：图像大小（默认为 256），在数据增强时会调整图像到这个大小。
# edge_decay：用于控制数据增强中的边缘裁剪的参数。
# only_h_flip：如果为 True，只进行水平翻转。
class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir) # 拼接得到最终的数据目录路径
		# 获取 'GT' 子文件夹中的所有文件名，并按字母顺序排序。 'GT' 存放了目标图像（ground truth images），
		# 这些图像将与源图像（'hazy' 文件夹）成对使用。
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

	def __len__(self): # 返回数据集中的图像对的数量
		return self.img_num
	# 关闭 OpenCV 的多线程和 OpenCL 加速，可能是为了避免多线程对某些环境或库的冲突
	# 确保数据加载时不使用并行加速
	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx] # 获取当前索引 idx 对应的图像文件名
		# 通过 read_img 函数读取源图像（假设是雾霾图像），并将其像素值缩放到 [-1, 1] 范围
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		# 读取目标图像（假设是清晰图像），并进行相同的像素缩放
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		# 如果当前模式是训练模式（'train'），对源图像和目标图像进行数据增强。增强包括裁剪、翻转、旋转等
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)
		# 如果模式是验证模式（'valid'），对源图像和目标图像进行对齐（裁剪到指定大小）
		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)
		# 返回一个字典，包含源图像、目标图像的转换后数据（'source' 和 'target'），以及当前图像的文件名（'filename'）
		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}

# 用于加载单张图像数据
class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
