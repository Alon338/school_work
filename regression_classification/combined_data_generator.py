#1
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def combined_data_generator(csv_file, images_dir, batch_size=32, target_size=(224, 224), shuffle=True):
    df = pd.read_csv(csv_file)
    # 分类标签存储为字符串如 "[1,0]"，用 eval 转换成 list
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
                # 处理分类标签
                if isinstance(row['class'], str):
                    # 存储的字符串为 "[1, 0]"，转换成 numpy array
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
