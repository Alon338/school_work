##4

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from python.data_enhancement import test_generator

# 从训练脚本保存的 best_model.h5 中加载已编译、已训练好的模型
model = tf.keras.models.load_model('best_model.h5')

# 在测试集上评估整体准确率和损失
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# 获取真实标签和预测标签
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 输出分类报告（包括精确率、召回率、F1 等指标）
target_names = ['Brain CT', 'Chest CT']
print(classification_report(y_true, y_pred, target_names=target_names))
