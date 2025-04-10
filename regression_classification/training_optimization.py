#4
import tensorflow as tf
import matplotlib.pyplot as plt
from regression_classification.combined_data_generator import combined_data_generator
from regression_classification.model_tensorFlow import combined_model
# 定义图像尺寸和批大小
img_height, img_width = 224, 224  # 输入模型的图像尺寸
#  CSV 文件路径
train_csv = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test\test_data.csv'
val_csv = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test\test_data.csv'
# 图片所在目录
images_dir = r'C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\test'

# 调用之前定义的自定义生成器
train_generator = combined_data_generator(train_csv, images_dir, batch_size=32, target_size=(img_height, img_width))
val_generator = combined_data_generator(val_csv, images_dir, batch_size=32, target_size=(img_height, img_width), shuffle=False)

# 计算每个 epoch 的步数：样本数量除以批量大小
train_steps = 100
val_steps = 20     # 根据验证集数量

# 设置回调函数：
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_combined_model.h5', save_best_only=True)
]

# 开始训练
history = combined_model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks
)

# 绘制训练过程中各任务的损失与评价指标曲线
# 绘制分类任务的准确率
plt.plot(history.history['classification_accuracy'], label='Train Classification Accuracy')
plt.plot(history.history['val_classification_accuracy'], label='Val Classification Accuracy')
plt.title('Classification Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制回归任务的 MAE 曲线
plt.plot(history.history['regression_mae'], label='Train Regression MAE')
plt.plot(history.history['val_regression_mae'], label='Val Regression MAE')
plt.title('Regression MAE vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
