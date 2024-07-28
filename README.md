# mlp_train_demo

MNIST训练集作为正样本(0-9)，CIFAR-100训练集作为负样本(negative)
实现手写数字识别

### [mlp_trainer.py](mlp_trainer.py)
训练模型并保存为onnx文件

### [cifar_process.py](cifar_process.py)
处理CIFAR-100，生成测试用png图片

### [mnist_process.py](mnist_process.py)
处理MNIST，生成测试用png图片

### [test_demo.py](test_demo.py)
使用OpenCV的dnn模块推理，测试模型训练效果