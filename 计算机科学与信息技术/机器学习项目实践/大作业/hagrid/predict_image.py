import argparse
import logging
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from constants import targets
from custom_utils.utils import build_model
import albumentations as A

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

def get_transform_for_inf(transform_config):
    transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)

def main():
    parser = argparse.ArgumentParser(description="单张图片分类预测")
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="配置文件路径")
    parser.add_argument("--img", required=True, type=str, help="图片路径")
    args = parser.parse_args()

    conf = OmegaConf.load(args.path_to_config)
    model = build_model(conf)
    transform = get_transform_for_inf(conf.test_transforms)
    if conf.model.checkpoint is not None:
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()

    img = cv2.imread(args.img)
    if img is None:
        print(f"图片读取失败: {args.img}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = transform(image=img_rgb)["image"]
    with torch.no_grad():
        output = model([processed])
    label = output["labels"].argmax(dim=1).item()
    print(f"预测类别: {targets[label]}")

if __name__ == "__main__":
    main()