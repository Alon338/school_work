import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from cladistic_taxonomy.data_enhancement import img_height, img_width

# -------------------- 重构模型（Functional API） --------------------
# 定义输入
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

# 构造预训练 VGG16 模型（不包含顶层全连接）
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False  # 冻结预训练模型

# 后续分类层
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation='softmax')(x)

# 构造完整模型
model = models.Model(inputs, outputs)
model.summary()

# -------------------- 构造 Grad-CAM 模型 --------------------
# 定义目标卷积层名称
last_conv_layer_name = 'block5_conv3'

# 直接从 base_model 中获取目标卷积层的输出
target_conv_output = base_model.get_layer(last_conv_layer_name).output

# 构造新的模型，输入与完整模型相同，输出为 [目标卷积层输出, 最终预测输出]
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[target_conv_output, model.output]
)

# -------------------- Grad-CAM 分析 --------------------
# 选择一张测试图像
img_path = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test\brain\007668.jpeg'
img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# 将 numpy 数组转换为 TensorFlow 张量
img_tensor = tf.convert_to_tensor(img_array)

# 使用 GradientTape 计算梯度
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    conv_outputs, predictions = grad_model(img_tensor)
    # 假设类别索引 0 表示脑部CT类别
    class_index = 0
    class_channel = predictions[:, class_index]

# 计算类别分数相对于目标卷积层输出的梯度
grads = tape.gradient(class_channel, conv_outputs)
if grads is None:
    raise ValueError("梯度计算失败，请检查模型的连通性。")
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# 去掉 batch 维度，得到形状 (H, W, channels)
conv_outputs = conv_outputs[0]

# 计算热力图：对每个通道，将该通道的特征图乘以对应的重要性权重，然后求和
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# 使用 ReLU 去除负值，并归一化到 [0,1]
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# 调整热力图尺寸至原图大小，并转换为 0-255 的 uint8 类型
heatmap = cv2.resize(heatmap, (img_width, img_height))
heatmap = np.uint8(255 * heatmap)

# 应用 JET 颜色映射生成彩色热力图
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 读取原始图像，并调整到相同尺寸
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (img_width, img_height))

# 将热力图与原图以一定透明度叠加
overlay_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# 保存叠加结果图
cv2.imwrite('gradcam_result.jpg', overlay_img)
