import streamlit as st
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from constants import targets
from custom_utils.utils import build_model
import albumentations as A
import numpy as np
from PIL import Image

# 图像增强变换 albumentations
def get_transform_for_inf(transform_config):
    transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
    # 图像转tensor
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)

# 缓存模型加载过程
@st.cache_resource
def load_model(config_path):
    conf = OmegaConf.load(config_path) # 加载配置
    model = build_model(conf) # 构建模型
    # 构建推理变换
    transform = get_transform_for_inf(conf.test_transforms)
    if conf.model.checkpoint is not None:
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu")) # 加载模型权重到CPU
        # 加载模型参数
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval() # 评估模式
    return model, transform

st.title("基于图像的手势识别")
config_path = st.text_input("请输入配置文件路径", "configs/MobileNetV3_large.yaml")
if config_path:
    model, transform = load_model(config_path) # 加载模型和变换
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 图片转RGB
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image) # 转numpy
        # 应用变换处理
        processed = transform(image=img_np)["image"]
        with torch.no_grad(): # 禁用梯度，仅推理
            output = model([processed]) # 前向推理
        # output标签最大概率的索引
        label = output["labels"].argmax(dim=1).item()
        st.image(image, caption="上传图片", use_column_width=True)
        st.success(f"预测类别: {targets[label]}")