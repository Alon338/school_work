#3,搭建 CNN 模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from image_classification.flow_from_directory import IMG_SIZE

model = Sequential()

# 卷积+池化 (可以多堆叠几层)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 展平
model.add(Flatten())

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 加个Dropout防止过拟合

# 输出层：二分类 => 1个神经元 + sigmoid
model.add(Dense(1, activation='sigmoid'))

model.summary()
