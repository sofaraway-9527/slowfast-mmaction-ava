import os

def list_filenames_in_directory(directory_path, output_file):
    # 打开输出文件
    with open(output_file, 'w') as file:
        # 遍历目录下的所有文件和子目录
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                # 将文件名写入输出文件
                file.write(filename + '\n')
    print(f"所有文件的文件名已保存到 {output_file}")

# 指定要遍历的目录路径和输出文件名
directory_path = 'yolovDeepsort/yolov5/runs/detect/exp/labels'  # 替换为你要遍历的目录路径
output_file = 'name.txt'  # 替换为你想要输出的文件名

# 调用函数
list_filenames_in_directory(directory_path, output_file)


