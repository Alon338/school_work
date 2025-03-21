##2

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

from python.data_enhancement import img_height, img_width

# 使用预训练的 VGG16 模型，不包含顶层全连接（include_top=False）
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(img_height, img_width, 3))
base_model.trainable = False  # 冻结预训练模型的卷积层参数

# 构建新模型（Sequential）
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),   # 将特征映射压缩为单一向量
    layers.Dense(256, activation='relu'),  # 全连接隐层，256 单元，ReLU 激活
    layers.Dropout(0.5),               # Dropout 正则化，防止过拟合
    layers.Dense(2, activation='softmax')  # 输出层，2 类，Softmax 激活
])
model.summary()  # 输出模型结构，确认各层名称和参数冻结情况

