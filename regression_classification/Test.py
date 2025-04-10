
from tensorflow.keras.models import load_model

# 加载模型（文件名可以是完整路径）
model = load_model(r'/regression_classification\best_combined_model.h5')

# 查看模型结构
model.summary()
