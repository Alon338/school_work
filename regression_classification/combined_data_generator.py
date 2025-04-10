#1

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def combined_data_generator(csv_file, images_dir, batch_size=32, target_size=(224, 224), shuffle=True):
    """
    自定义数据生成器，同时输出两个任务的标签：分类标签和回归目标

    参数：
      csv_file: 包含数据的信息文件路径（CSV），必须包含 'filename'、'class'、'regression' 三列
      images_dir: 存放图片的目录
      batch_size: 每批次加载的样本数量
      target_size: 图片调整到的大小 (height, width)
      shuffle: 是否在每个epoch打乱数据顺序

    返回：
      每次返回一个batch的样本和对应标签：
         x: 图像数组，归一化到 [0,1]
         y: 一个字典，包含键 'classification' 与 'regression'
    """
    df = pd.read_csv(csv_file)
    # 若分类标签存储为字符串如 "[1,0]"，可以用 eval 转换成 list；若直接是整数，则可能需要转换为 one-hot 编码
    while True:
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]
            images = []
            class_labels = []
            regression_labels = []
            for _, row in batch_df.iterrows():
                file_path = os.path.join(images_dir, row['filename'])
                # 加载并调整图像大小
                img = load_img(file_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0  # 归一化
                images.append(img_array)

                # 处理分类标签：这里假定如果为字符串则转换为列表；否则直接取值
                if isinstance(row['class'], str):
                    # 示例：如果存储的字符串为 "[1, 0]"，则转换成 numpy array
                    label = np.array(eval(row['class']))
                else:
                    label = row['class']
                class_labels.append(label)

                regression_labels.append(row['regression'])

            images = np.array(images)
            class_labels = np.array(class_labels)
            # 保证回归标签为 (batch_size, 1) 的形状
            regression_labels = np.array(regression_labels).reshape(-1, 1)
            yield images, {'classification': class_labels, 'regression': regression_labels}
