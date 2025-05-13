from preprocess import process_and_infer
from predict import generate_digit
import os

if __name__ == "__main__":
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取所在目录
    output_path = os.path.dirname(script_path)

    # 预处理和推理
    model_path = 'cnn_model_big.pth'
    img_path = 'test.jpg'
    predicted_class = process_and_infer(img_path, model_path)
    print(f"The predicted class is: {predicted_class}")

    # 生成数字
    generate_digit(digit=predicted_class, model_path="cvae_model.pt", output_path=output_path, num_samples=6)
