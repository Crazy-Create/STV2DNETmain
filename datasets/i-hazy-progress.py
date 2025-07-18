#将I-HAZY文件下的图片处理成和Indooor一样的格式

import os
import re

# 原始路径配置
base_dir = r'D:\Pycharm\DehazeFormer-test2\data\I-HAZY\test\hazy'
#base_dir = r'D:\Pycharm\DehazeFormer-test2\data\I-HAZY\test\GT'
for filename in os.listdir(base_dir):
    if filename.lower().endswith('.jpg'):
        # 使用正则提取文件名开头的连续数字[1,3](@ref)
        num_match = re.match(r'^(\d+)', filename)

        if num_match:
            # 构造新文件名（保留前导零）[3,8](@ref)
            new_name = f"{num_match.group(1)}.jpg"

            # 生成完整路径
            old_path = os.path.join(base_dir, filename)
            new_path = os.path.join(base_dir, new_name)

            # 执行重命名[7](@ref)
            os.rename(old_path, new_path)
            print(f"重命名完成: {filename} -> {new_name}")

print("批量重命名操作完成！")