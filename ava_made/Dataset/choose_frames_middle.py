import os
import shutil

# 遍历 ./choose_frames 目录
for filepath, dirnames, filenames in os.walk(r'./choose_frames'):
    if len(filenames) == 0:
        continue
    
    # 在 choose_frames_middle 下创建对应的目录文件夹
    temp_name = os.path.basename(filepath)  # 获取目录名
    path_temp_name = os.path.join('./choose_frames_middle', temp_name)
    
    if not os.path.exists(path_temp_name):
        os.makedirs(path_temp_name)
        print(f"Created directory: {path_temp_name}")
    
    filenames = sorted(filenames)
    
    # 找到指定的图片，然后移动到 choose_frames 中对应的文件夹下
    for filename in filenames:
        if "checkpoint" in filename or "Store" in filename:
            continue
        
        temp_num = filename.split('_')[1]
        temp_num = temp_num.split('.')[0]
        temp_num = int(temp_num)
        
        if (temp_num - 1) / 30 <= 1 or (temp_num - 1) / 30 >= len(filenames) - 2:
            continue
        
        temp_num_str = str(temp_num).zfill(6)
        new_filename = f"{temp_name}_{temp_num_str}.jpg"
        
        srcfile = os.path.join(filepath, new_filename)
        dstpath = os.path.join(path_temp_name, new_filename)
        
        # 检查源文件是否存在
        if os.path.exists(srcfile):
            # 复制文件
            shutil.copy(srcfile, dstpath)
            print(f"Copied: {srcfile} to {dstpath}")
        else:
            print(f"Source file does not exist: {srcfile}")
