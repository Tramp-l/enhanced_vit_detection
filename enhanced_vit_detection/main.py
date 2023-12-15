import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from enhanced_vit import EnhancedViT
from datasets import CustomDataset
from adversarial_detector import AdversarialSampleDetector


def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 加载数据集
    train_dataset = CustomDataset("train_dataset_path")
    test_dataset = CustomDataset("test_dataset_path")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化增强的ViT模型
    enhanced_vit = EnhancedViT(num_classes=10)

    # 初始化对抗样本检测器
    detector = AdversarialSampleDetector(enhanced_vit)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(enhanced_vit.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(num_epochs):
        enhanced_vit.train()
        for images, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = enhanced_vit(images)
            loss = criterion(outputs, labels)

            # 使用对抗样本检测器检测对抗样本
            pixel_values = images  # 假设您的数据已经准备好
            mse_loss = detector(pixel_values)
            z_score_threshold = 3  # 根据您的需求设置阈值
            adversarial_detected = detector.detect(mse_loss, z_score_threshold)

            # 如果检测到对抗样本，不更新模型参数
            if not adversarial_detected.any():
                loss.backward()
                optimizer.step()

        # 在每个epoch结束后进行测试
        enhanced_vit.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                outputs = enhanced_vit(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}] Test Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    num_epochs = 10  # 设置训练的epoch次数
    main()
