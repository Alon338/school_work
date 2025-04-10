#2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# 定义输入图像尺寸与类别数量
img_height, img_width = 224, 224
num_classes = 2  # 类别数量

# ---- 1. 定义输入层 ----
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

# ---- 2. 构建预训练基础模型 ----
# 使用 VGG16 预训练模型作为特征提取器
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False  # 冻结基础模型参数，避免在初期训练时发生过拟合

# ---- 3. 添加共享层 ----
# 对基础模型输出做全局平均池化，将空间特征图转换为固定长度向量
x = layers.GlobalAveragePooling2D()(base_model.output)
# 添加全连接层进一步提取特征
x = layers.Dense(256, activation='relu')(x)
# 采用 Dropout 防止过拟合
x = layers.Dropout(0.5)(x)

# ---- 4. 构建分类分支 ----
# 分类分支的输出层采用 softmax 激活，输出类别概率
cls_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)

# ---- 5. 构建回归分支 ----
# 回归分支的输出层采用线性激活，用于预测连续数值
reg_output = layers.Dense(1, activation='linear', name='regression')(x)

# ---- 6. 构造多输出模型 ----
combined_model = models.Model(inputs=inputs, outputs=[cls_output, reg_output])

# ---- 7. 编译模型 ----
# 同时指定两个任务的损失函数、损失权重与评价指标
combined_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={'classification': 'categorical_crossentropy', 'regression': 'mean_squared_error'},
    loss_weights={'classification': 1.0, 'regression': 0.5},  # 回归和分类的权重调整（可以通过调整权重确保模型两边任务平衡学习）
    metrics={'classification': 'accuracy', 'regression': 'mae'}
)

combined_model.summary()  # 输出模型结构，方便检查网络连接和参数数量
