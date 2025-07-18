import numpy as np
import cv2

# 计算和存储数值的类，目的是帮助计算在一段时间内数值（如损失、准确率等）的平均值、总和和当前值
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

# 处理一个值的列表。len 初始化为 10000，意味着默认最多处理 10000 个值。
class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			
# 函数用于读取图像并进行预处理
def read_img(filename):
	img = cv2.imread(filename)
	return img[:, :, ::-1].astype('float32') / 255.0 # 将像素值缩放到 [0, 1] 的范围


def write_img(filename, img):  # 将图像保存到文件
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)


def hwc_to_chw(img):  # 将图像的维度从 HWC（Height, Width, Channel）转换为 CHW（Channel, Height, Width）
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):  # 将图像的维度从 CHW（Channel, Height, Width）转换回 HWC（Height, Width, Channel）
	return np.transpose(img, axes=[1, 2, 0]).copy()
