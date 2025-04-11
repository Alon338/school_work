#1,数据
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 图片缩放到224×224
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 设定数据路径
TRAIN_DIR = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data/train'
VAL_DIR = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data/val'
TEST_DIR = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data/test'

# 定义用于训练的数据增广器
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 将像素值缩放到 [0,1]
    rotation_range=10,       # 随机旋转 ±10度
    width_shift_range=0.1,   # 随机水平平移
    height_shift_range=0.1,  # 随机垂直平移
    zoom_range=0.1,          # 随机缩放
    horizontal_flip=True     # 随机水平翻转
)

# 验证和测试通常只做归一化，不做增强
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 从文件夹读取训练/验证/测试数据
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # 二分类
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    directory=VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# 打印类别索引映射：{"brain": 0, "chest": 1}
print("Class indices:", train_generator.class_indices)
