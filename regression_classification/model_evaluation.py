#3
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
import pandas as pd

from regression_classification.combined_data_generator import combined_data_generator
from regression_classification.training_optimization import images_dir
# 定义图像尺寸和批大小
img_height, img_width = 224, 224  # 输入模型的图像尺寸
# 加载保存的最优模型
combined_model = tf.keras.models.load_model('best_combined_model.h5')

# 构造测试集生成器
test_csv = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test\test_data.csv'
test_generator = combined_data_generator(test_csv, images_dir, batch_size=32, target_size=(img_height, img_width), shuffle=False)
# 测试步数：根据你的测试集大小设置
test_steps = 20

# 在测试集上评估整体损失与指标
results = combined_model.evaluate(test_generator, steps=test_steps)
print("Test loss and metrics:", results)

# 对测试集进行预测
predictions = combined_model.predict(test_generator, steps=test_steps)
# predictions 是一个列表，第一个元素为分类分支预测，第二个为回归分支预测

# 处理分类任务
cls_pred_prob = predictions[0]   # 分类预测概率
cls_predictions = np.argmax(cls_pred_prob, axis=1)
df_test = pd.read_csv(test_csv)
# 把 one-hot（字符串）转换为数值标签
y_true_cls = np.array([np.argmax(np.array(eval(x))) if isinstance(x, str) else x for x in df_test['class']])

# 计算并输出混淆矩阵和分类报告
cm = confusion_matrix(y_true_cls, cls_predictions)
print("Confusion Matrix:")
print(cm)

target_names = ['brain', 'chest']  # 分类标签名称
print("Classification Report:")
print(classification_report(y_true_cls, cls_predictions, target_names=target_names))

# 处理回归任务
reg_pred = predictions[1].flatten()  # 回归预测结果
# 从 CSV 文件中获取真实的回归目标
y_true_reg = df_test['regression'].values.astype(np.float32)

# 计算回归指标
mse = mean_squared_error(y_true_reg, reg_pred)
mae = mean_absolute_error(y_true_reg, reg_pred)
print(f"Regression MSE: {mse:.4f}, MAE: {mae:.4f}")
