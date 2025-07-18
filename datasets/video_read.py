
#1. photo_name_1--------------------------------------------------------------------------------------------------------------
#将指定文件路径下的图片不管图片命名为什么例如（sfrg11423bfgdh.jpg），都统一从1开始排序命名例如（1.jpg,2.jpg...70.jpg）
import os
def photo_name_1():
    # 指定文件路径
    directory = r"C:/Users/ASUS/Desktop/检修库"
    # 获取所有jpg文件并排序
    files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    # 重命名文件
    for index, filename in enumerate(files, start=1201):    #设置起始命名序号
        old_file_path = os.path.join(directory, filename)
        new_file_name = f"{index}.jpg"
        new_file_path = os.path.join(directory, new_file_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)

    print("文件重命名完成。")

#2. photo_resize_2-------------------------------------------------------------------------------
# 读取文件路径下的图片，并处理为1280*720后，保存到photo2文件夹
import os
import cv2
def photo_resize_2():

    # 输入和输出图像路径
    input_folder = 'C:/Users/ASUS/Desktop/zhengbeichang100'  # 输入文件夹路径
    output_folder = 'C:/Users/ASUS/Desktop/zhengbeichangw100'  # 输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 处理每个图像文件
    for image_name in image_files:
        input_image_path = os.path.join(input_folder, image_name)
        output_image_path = os.path.join(output_folder, image_name)

        # 检查输入图像是否存在
        if not os.path.exists(input_image_path):
            print(f"图像文件不存在: {input_image_path}")
            continue
        # else:
        #     #print(f"图像文件已找到: {input_image_path}")

        # 加载图像
        image = cv2.imread(input_image_path)

        # 获取原始图像的尺寸
        height, width = image.shape[:2]

        # 定义目标尺寸
        target_width = 1280
        target_height = 720

        # 计算缩放比例
        scale_width = target_width / float(width)
        scale_height = target_height / float(height)

        # 选择较小的比例，保证等比缩放
        scale = min(scale_width, scale_height)

        # 计算新的尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 缩放图像
        resized_image = cv2.resize(image, (new_width, new_height))

        # 填充图像至目标尺寸（如果需要填充）
        top = (target_height - new_height) // 2
        bottom = target_height - new_height - top
        left = (target_width - new_width) // 2
        right = target_width - new_width - left

        # 使用均匀颜色填充（此处填充为黑色）
        output_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 保存处理后的图像
        cv2.imwrite(output_image_path, output_image)
        print(f"图像已保存到 {output_image_path}")


#3. video_read_photo_3()----------------------------------------------------------------------------------
#读取视频流video.mp4,每格2秒读取一张图片，将读取出来的图片按照video_1、2...保存至photo4文件夹.视频为1280*720，输出图片大小为1280*720
import cv2
import os
def video_read_photo_3(secend):
    # 输入视频文件路径
    #video_file = '../data/video3/video17_test.mp4'
    video_file = "D:/Pycharm/dd.mp4"

    # 输出文件夹路径
    output_folder = 'C:/Users/ASUS/Desktop/dataset/'
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()
    # 获取视频的帧率（每秒的帧数）
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 初始化帧计数器
    frame_number = 0
    # 读取视频流
    while True:
        # 设置视频流读取位置为当前帧的秒数位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number * fps * secend)  # 每隔2秒读取一帧
        # 读取帧
        ret, frame = cap.read()
        # 如果没有帧了，结束循环
        if not ret:
            break

        # 保存当前帧为图片
        output_image_path = os.path.join(output_folder, f"video24_{frame_number + 1}.jpg")
        cv2.imwrite(output_image_path, frame)
        print(f"保存图像: {output_image_path}")
        # 增加帧计数器（每次增加1秒）
        frame_number += 1
    # 释放视频捕获对象
    cap.release()


# 4.fog_photo_4()在photo4文件夹下有很多图片都是720*1080的图片，将他们处理成带雾的图像，
# 每一张图像随机生成1张带雾的图像命名格式为:输入图像名字_1 （Place1_1）将其保存至photo3文件夹

import os
import cv2
import random
import math
import numpy as np

def fog_photo_4():
    # 输入和输出文件夹路径
    folder_path = '../data/photo4/video16'  # 输入图片文件夹路径
    fog_path = '../data/photo5/video16'  # 输出图片文件夹路径

    # 如果输出文件夹不存在，创建该文件夹
    if not os.path.exists(fog_path):
        os.makedirs(fog_path)

    # 遍历文件夹中的每个图片文件
    for file_img in os.listdir(folder_path):
        # 拼接文件路径
        img_path = os.path.join(folder_path, file_img)

        # 只处理图片文件，可以通过文件扩展名来过滤
        if file_img.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)  # 读取原图
            mask_img = img.copy()  # 创建雾霾图
            # 归一化并应用雾霾模型
            img = img / 255.0  # 归一化到[0, 1]
            (row, col, chs) = img.shape

            A = random.uniform(0.6, 1)  # 亮度
            beta = random.uniform(0.004, 0.09)  # 雾的浓度0.09 - 0.004
            size = math.sqrt(max(row, col))  # 雾化尺寸
            center = (row // 2, col // 2)  # 雾化中心
            center_row = random.randint(row // 3, int(row * 0.7))
            center_col = random.randint(col // 3, int(col * 0.7))
            center = (center_row, center_col)

            # 使用广播和向量化操作代替逐像素计算
            Y, X = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')  # 创建Y, X坐标网格
            dist = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)  # 计算每个像素到中心的距离

            d = -0.04 * dist + size  # 计算每个像素的d值
            td = np.exp(-beta * d)  # 计算透过率

            # 生成带雾霾的图像
            mask_img = img * td[..., np.newaxis] + A * (1 - td[..., np.newaxis])  # 广播应用模型

            # 将图像恢复到[0, 255]范围
            mask_img = np.clip(mask_img * 255, 0, 255).astype(np.uint8)

            # 拼接输出路径，并保存文件
            output_path = os.path.join(fog_path, file_img)
            cv2.imwrite(output_path, mask_img)  # 保存带雾霾的图片
            print(f"保存文件: {output_path}")


# 5.copy_1200_5(),将同一张照片复制1200张，图片大小不变
import shutil
import os
def copy_1200_5():
    source_image = '../data/RESIDE-OUT/test/video17_GT/17.jpg'  # 请根据实际情况调整文件后缀
    # 目标文件夹路径
    target_folder = '../data/RESIDE-OUT/test/video17_GT'
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # 复制1200次
    for i in range(1, 1201):
        # 构建新的文件名
        new_image_name = f"video17_{i}.jpg"  # 名字格式 video17_1.jpg, video17_2.jpg ...
        # 复制文件到目标文件夹
        target_path = os.path.join(target_folder, new_image_name)
        shutil.copy(source_image, target_path)
    print("图片复制完成！")


if __name__ == '__main__':
    # 1.将指定文件路径photo1下的图片不管图片命名为什么例如（sfrg11423bfgdh.jpg），都统一从1开始排序命名例如（1.jpg,2.jpg...70.jpg）
     photo_name_1()
    # 2.读取文件路径下的图片，并处理为1280*720后，保存到photo3文件夹
    # photo_resize_2()
    # 3.读取视频流video.mp4,每格3秒读取一张图片，将读取出来的图片按照video_1、2...保存至photo4文件夹.视频为1280*720，输出图片大小为1280*720
    # video_read_photo_3(secend=2)
    # 4.fog_photo_4()在photo4文件夹下有很多图片都是720*1080的图片，将他们处理成带雾的图像
    # fog_photo_4()
    # 5.将同一张照片复制1200张，图片大小不变
     #copy_1200_5()
