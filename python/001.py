##1

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义图像尺寸和批大小
img_height, img_width = 224, 224  # 输入模型的图像尺寸
batch_size = 32

# 定义训练数据的增强配置（包括随机变换和归一化）
train_datagen = ImageDataGenerator(
    rescale=1.0/255,         # 将像素值归一化到[0,1]
    rotation_range=40,       # 随机旋转0~40度&#8203;:contentReference[oaicite:3]{index=3}
    width_shift_range=0.2,   # 水平平移±20%宽度
    height_shift_range=0.2,  # 垂直平移±20%高度
    zoom_range=0.2,          # 随机缩放±20%
    horizontal_flip=True,    # 随机水平翻转
    fill_mode='nearest'      # 填充像素采用最近邻法
)
# 验证集和测试集一般只做归一化，不做额外增强
test_datagen = ImageDataGenerator(rescale=1.0/255)

# 从目录读取训练集（shuffle=True 会打乱数据顺序）
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\ASUS\IdeaProjects\VisualAnalysis/data/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # 二分类可用 'binary'; 这里用2个类别的categorical
)
# 从目录读取验证集
val_generator = test_datagen.flow_from_directory(
    'C:\\Users\ASUS\IdeaProjects\VisualAnalysis/data/val',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # 验证/测试集不需要乱序，以保证评估准确性
)
# 从目录读取测试集
test_generator = test_datagen.flow_from_directory(
    'C:\\Users\ASUS\IdeaProjects\VisualAnalysis/data/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)



##2

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

from python.data_enhancement import img_height, img_width

# 使用预训练的VGG16模型，不包含顶层全连接（include_top=False）
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(img_height, img_width, 3))
base_model.trainable = False  # 冻结预训练模型的卷积层参数

# 构建新模型
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),           # 将特征映射压缩为单一矢量
    layers.Dense(256, activation='relu'),      # 全连接隐层，256单元，ReLU激活
    layers.Dropout(0.5),                       # dropout正则化，防止过拟合
    layers.Dense(2, activation='softmax')      # 输出层，2类，Softmax激活
])
model.summary()  # 输出模型结构概要，确认模型各层名称和参数冻结情况


##3

#编译模型
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from python.data_enhancement import train_generator, val_generator
from python.model_building import model

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),  # Adam优化器，学习率可调整
    loss='categorical_crossentropy',               # 交叉熵损失（适用于Softmax输出）
    metrics=['accuracy']                           # 评估指标：准确率
)
#模型训练
# 可选：定义早停和模型检查点回调
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

# 训练过程中，每个epoch会输出训练集和验证集的损失(loss)和准确率(accuracy)。通过history对象
# 绘制训练准确率和验证准确率曲线
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制训练损失和验证损失曲线
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


##4,model_evaluation

# 在测试集上评估整体准确率和损失
from python.data_enhancement import test_generator
from python.model_building import model

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# 计算测试集的混淆矩阵
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 获取真实标签和预测标签
y_true = test_generator.classes  # 测试集真实标签 (0或1)
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)  # 概率最大的位置作为预测类别

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# 输出每类的查准率、召回率、F1分数等
target_names = ['Brain CT', 'Chest CT']
print(classification_report(y_true, y_pred, target_names=target_names))


##5
import cv2
import tensorflow as tf
import numpy as np
from python.data_enhancement import img_height, img_width
from python.model_building import model

# ------------------- Grad-CAM 部分 -------------------
# 定义目标卷积层名称
last_conv_layer_name = 'block5_conv3'

# 从顶层模型中获取 VGG16 子模型（在 Sequential 模型中通常为第一层）
vgg16_model = model.get_layer("vgg16")

# 从 VGG16 子模型内部获取目标卷积层的输出
last_conv_output = vgg16_model.get_layer(last_conv_layer_name).output

# 构造用于 Grad-CAM 的新模型：
# 使用 vgg16_model.input 作为输入，确保计算图连通
# 输出为目标卷积层输出和整个模型的最终预测输出
grad_model = tf.keras.models.Model(
    inputs=vgg16_model.input,
    outputs=[last_conv_output, model.output]
)

# 选择一张测试图像进行 Grad-CAM 分析
img_path = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test\brain\007668.jpeg'
img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # 与训练时保持一致的预处理

# 使用 GradientTape 计算梯度：提前跟踪输入张量以确保梯度正确计算
with tf.GradientTape() as tape:
    tape.watch(img_array)
    conv_outputs, predictions = grad_model(img_array)
    # 假设索引 0 为脑部CT类别
    class_index = 0
    class_channel = predictions[:, class_index]

# 计算类别分数相对于卷积层输出的梯度
grads = tape.gradient(class_channel, conv_outputs)

# 计算每个特征通道梯度的全局平均值，作为通道的重要性权重
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# 去掉 batch 维度，conv_outputs 的 shape 为 (H, W, channels)
conv_outputs = conv_outputs[0]

# 计算热力图：将每个特征通道的特征图乘以对应的重要性权重后求和
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# 通过 ReLU 去除负值，并归一化到 [0,1]
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# 调整热力图尺寸至原图大小，并转换为 0-255 的 uint8 类型
heatmap = cv2.resize(heatmap, (img_width, img_height))
heatmap = np.uint8(255 * heatmap)

# 应用 JET 颜色映射生成彩色热力图
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 读取原始图像，并将热力图与原图按一定透明度叠加
original_img = cv2.imread(img_path)
overlay_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# 保存叠加结果图
cv2.imwrite('gradcam_result.jpg', overlay_img)

