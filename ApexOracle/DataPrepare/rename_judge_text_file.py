import os

# 替换成你的文件夹路径
folder_path = '/data2/tianang/projects/Synergy/DataPrepare/Data/Text_Description/wo_ATCC/judge_exist_text'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 只处理文件，不处理子文件夹
    old_path = os.path.join(folder_path, filename)
    if os.path.isfile(old_path):
        # 如果文件名中包含下划线，则进行替换
        if '_' in filename:
            new_filename = filename.replace('_', '～')
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f'重命名: {filename} → {new_filename}')