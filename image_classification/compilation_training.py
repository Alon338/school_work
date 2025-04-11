#2,编译与训练
from image_classification.cnn_model import model
from image_classification.flow_from_directory import train_generator,val_generator

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=1e-4)  # 尝试较低的学习率
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

EPOCHS = 50

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)
