#4,测试与评估
from image_classification.cnn_model import model
from image_classification.flow_from_directory import test_generator
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',    # 二分类用binary_crossentropy
    metrics=['accuracy']
)

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
