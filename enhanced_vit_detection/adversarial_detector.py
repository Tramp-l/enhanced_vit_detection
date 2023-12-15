import torch
import torch.nn as nn
import torch.nn.functional as F
from enhanced_vit import EnhancedViT
from tv_denoise import TVDenoise #tv_denoise

class AdversarialSampleDetector(nn.Module):
    def __init__(self, enhanced_vit):
        super(AdversarialSampleDetector, self).__init__()
        self.enhanced_vit = EnhancedViT #enhanced_vit
        self.tv_weight = 0.1 # Total Variation weight

    def forward(self, pixel_values):
        # 应用增强的ViT模型来提取特征
        logits = self.enhanced_vit(pixel_values)
        predictions = F.softmax(logits, dim=-1)

        # 应用总变差(TV)降噪
        denoised_images = TVDenoise(pixel_values)

        # 使用增强的ViT模型再次提取降噪后的特征
        denoised_logits = self.enhanced_vit(denoised_images)
        denoised_predictions = F.softmax(denoised_logits, dim=-1)

        # 计算原始和降噪后的损失均方误差
        mse_loss = F.mse_loss(predictions, denoised_predictions, reduction='none')
        mse_loss = mse_loss.mean(1)

        # 返回MSE损失，用于后续的Z-score计算和异常检测
        return mse_loss

    def detect(self, mse_loss, z_score_threshold):
        # 计算Z-score
        mean = mse_loss.mean()
        std = mse_loss.std()
        z_scores = (mse_loss - mean) / std

        # 检测高于阈值的Z-score，认为它们是对抗样本
        detected_adversaries = z_scores > z_score_threshold
        return detected_adversaries