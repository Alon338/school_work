import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from image_classification.flow_from_directory import IMG_SIZE
from image_classification.cnn_model import model

def predict_single_image(image_path):
    """
    加载单张CT图像，根据训练好的CNN模型判断图片是 chest 还是 brain。
    参数:
      - image_path: CT图像文件的路径
    """
    # 加载图像，并调整尺寸为训练时使用的 IMG_SIZE
    img = load_img(image_path, target_size=IMG_SIZE)
    # 转换为数组，并归一化处理，使像素值范围与训练时一致
    x = img_to_array(img) / 255.0
    # 扩展维度，变为 (1, height, width, channels)
    x = np.expand_dims(x, axis=0)

    # 利用训练好的模型进行预测，返回一个概率值数组，如 [[0.32]] 或 [[0.78]]
    prediction = model.predict(x)
    print("模型预测概率：", prediction[0][0])

    # 根据阈值0.5判断类别（请确保这一设定与你的数据和类别编码一致）
    if prediction[0][0] < 0.5:
        print("预测结果：brain")
    else:
        print("预测结果：chest")

if __name__ == "__main__":
    # 指定待测试CT图像的路径
    # image_path = r"C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\val\brain\007768.jpeg"
    image_path = r"C:\Users\ASUS\IdeaProjects\VisualAnalysis\data\val\chest\007520.jpeg"
    predict_single_image(image_path)
