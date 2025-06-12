import argparse
from typing import Optional, Tuple

from omegaconf import OmegaConf
from torch.distributed import destroy_process_group

# 平均精度均值
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP

    MeanAveragePrecision = MAP

from custom_utils.ddp_utils import ddp_setup
from custom_utils.train_utils import Trainer, load_train_objects, load_train_optimizer
from custom_utils.utils import F1ScoreWithLogging

def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手势分类器...")
    parser.add_argument(
        "-c", "--command", required=True, type=str, help="训练/测试", choices=("train", "test")
    )
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="配置文件路径")
    parser.add_argument("--n_gpu", required=False, type=int, default=1, help="使用的GPU数量")
    known_args, _ = parser.parse_known_args(params)
    return known_args

def run(args):
    config = OmegaConf.load(args.path_to_config)
    if args.n_gpu > 1:
        ddp_setup()
    # 加载数据集、模型
    train_dataloader, val_dataloader, test_dataloader, model = load_train_objects(config, args.command, args.n_gpu)
    # 根据模型类型选择评价指标
    if model.type == "detector": # 检测模型 SSD/YOLO
        metric = MeanAveragePrecision(class_metrics=False) # mAP
    else: # 分类模型 MobileNetV3/ResNet
        task = "binary" if config.dataset.one_class else "multiclass" # 二分类/多分类
        num_classes = 2 if config.dataset.one_class else len(config.dataset.targets) # 类别数
        metric = F1ScoreWithLogging(task=task, num_classes=num_classes) # F1Score
        # F1Score = 2 * (precision * recall) / (precision + recall)
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # TP 预测为正，实际为正 FP 预测为正，实际为负 FN 预测为负，实际为正
    # 优化器、学习率调度器
    optimizer, scheduler = load_train_optimizer(model, config)
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_calculator=metric,
        train_data=train_dataloader,
        val_data=val_dataloader,
        test_data=test_dataloader,
        n_gpu=args.n_gpu,
    )

    if args.command == "train":
        trainer.train()
    if args.command == "test":
        trainer.test()

    if args.n_gpu > 1:
        destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
