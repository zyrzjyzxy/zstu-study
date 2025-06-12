import argparse
import logging
import time
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0) # 绿色
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        图像预处理，将输入图片进行增强和转换为模型输入格式
        参数
        ----------
        img: np.ndarray
            输入图片
        transform :
            albumentations增强操作
        """
        transformed_image = transform(image=img)
        return transformed_image["image"]

    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        """
        根据配置生成推理时的增强操作列表
        参数
        ----------
        transform_config: DictConfig
            测试增强配置
        """
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        # 添加Tensor转换
        transforms_list.append(ToTensorV2())
        # 返回组合增强
        return A.Compose(transforms_list)

    @staticmethod
    def run(classifier, transform) -> None:
        """
        运行分类模型并在摄像头画面上显示类别标签和FPS
        参数
        ----------
        classifier : TorchVisionModel
            分类模型
        transform :
            albumentations增强操作
        """

        cap = cv2.VideoCapture(0) # 打开摄像头
        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1 # 计算帧间隔
            t1 = time.time()

            ret, frame = cap.read() # 读取一帧
            if ret:
                # 预处理
                processed_frame = Demo.preprocess(frame, transform)
                with torch.no_grad():
                    # 推理
                    output = classifier([processed_frame])
                # 获取预测类别
                label = output["labels"].argmax(dim=1)

                cv2.putText(  # 显示类别标签
                    frame, targets[int(label)], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
                )
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, (255, 0, 255), 2)
                cnt += 1

                cv2.imshow("Frame", frame) # 显示帧

                key = cv2.waitKey(1)
                if key == ord("q"): # 按q退出
                    return
            else:
                cap.release()
                cv2.destroyAllWindows() # 关闭窗口


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Demo full frame classification...")

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()  # 解析命令行参数
    conf = OmegaConf.load(args.path_to_config) # 加载配置文件
    model = build_model(conf) # 构建模型
    transform = Demo.get_transform_for_inf(conf.test_transforms) # 获取测试增强
    if conf.model.checkpoint is not None:
        # 加载权重
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval() # 设置为评估模式
    if model is not None:
        Demo.run(model, transform) # 启动摄像头推理
