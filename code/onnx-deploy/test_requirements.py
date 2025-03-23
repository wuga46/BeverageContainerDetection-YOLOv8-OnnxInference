# 如果Pytorch已经安装，请忽略下一步
# pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# 安装工具
# pip install numpy pandas matplotlib tqdm opencv-python pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装onnx和onnxruntime
# pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

import onnx
print('ONNX 版本', onnx.__version__)

import onnxruntime as ort
print('ONNX Runtime 版本', ort.__version__)
