##1

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义图像尺寸和批大小
img_height, img_width = 224, 224  # 输入模型的图像尺寸
batch_size = 32

# 定义训练数据的增强配置（包括随机变换和归一化）
train_datagen = ImageDataGenerator(
    rescale=1.0/255,         # 将像素值归一化到 [0,1]
    rotation_range=40,       # 随机旋转 0~40 度
    width_shift_range=0.2,   # 水平平移 ±20% 宽度
    height_shift_range=0.2,  # 垂直平移 ±20% 高度
    zoom_range=0.2,          # 随机缩放 ±20%
    horizontal_flip=True,    # 随机水平翻转
    fill_mode='nearest'      # 填充像素采用最近邻法
)

# 验证集和测试集一般只做归一化，不做额外增强
test_datagen = ImageDataGenerator(rescale=1.0/255)

# 从目录读取训练集（建议使用原始字符串）
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
# 从目录读取验证集
val_generator = test_datagen.flow_from_directory(
    r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
# 从目录读取测试集
test_generator = test_datagen.flow_from_directory(
    r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
