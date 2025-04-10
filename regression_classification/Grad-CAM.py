import cv2
import numpy as np
import tensorflow as tf
from regression_classification.model_tensorFlow import combined_model,base_model
# 定义图像尺寸和批大小
img_height, img_width = 224, 224  # 输入模型的图像尺寸
# 假设我们仍使用 VGG16 中的最后一个卷积层 'block5_conv3' 作为目标
last_conv_layer = base_model.get_layer('block5_conv3')
# 构造Grad-CAM模型，输出为目标卷积层输出和分类分支的预测
grad_model = tf.keras.models.Model(
    inputs=combined_model.input,
    outputs=[last_conv_layer.output, combined_model.get_layer('classification').output]
)

# 选取一张测试图像进行分析
img_path = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test/chest/007404.jpeg'
img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
img_tensor = tf.convert_to_tensor(img_array)

with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    conv_outputs, predictions = grad_model(img_tensor)
    # 假定我们关注类别索引 0
    class_index = 0
    class_channel = predictions[:, class_index]

grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

heatmap = cv2.resize(heatmap, (img_width, img_height))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (img_width, img_height))
overlay_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
cv2.imwrite('gradcam_combined_result.jpg', overlay_img)
