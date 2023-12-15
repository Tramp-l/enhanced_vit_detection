import torch
from torch import nn
from transformers import  ViTModel


class EnhancedViT(nn.Module):
    def __init__(self, num_classes, image_size, num_alpha_heads, num_beta_heads):
        super(EnhancedViT, self).__init__()

        # 初始化基础的ViT模型
        self.base_vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # 获取ViT模型的隐藏层维度
        hidden_dim = self.base_vit.head.in_features

        # α 注意力头
        self.alpha_attention_heads = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_alpha_heads)

        # β 注意力头
        self.beta_attention_heads = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_beta_heads)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # 用于合并两个注意力头的输出
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # 将图像通过基础ViT模型
        vit_outputs = self.base_vit(x)

        # 提取transformer的最后一层输出
        last_layer_output = vit_outputs.last_hidden_state

        # 将特征调整为合适的形状以用于多头注意力
        transformer_features = last_layer_output.permute(1, 0, 2)

        # 计算α注意力头
        alpha_attention_output, _ = self.alpha_attention_heads(transformer_features, transformer_features,
                                                               transformer_features)

        # 计算β注意力头
        beta_attention_output, _ = self.beta_attention_heads(transformer_features, transformer_features,
                                                             transformer_features)

        # 合并两个头的输出
        combined_output = torch.cat((alpha_attention_output, beta_attention_output), dim=2)

        # 序列池化，将所有注意力输出的平均值作为最终特征
        pooled_output = combined_output.mean(dim=0)

        # 通过分类器得到最终的输出
        logits = self.classifier(pooled_output)

        return logits


# 初始化模型
model = EnhancedViT(
    num_classes=10,  # CIFAR-10的类别数
    image_size=(224, 224),  # 输入图像大小
    num_alpha_heads=8,  # α 注意力头的数量
    num_beta_heads=8  # β 注意力头的数量
)

# 输入数据 [batch_size, channels, height, width]
input_tensor = torch.randn(32, 3, 224, 224)

# 前向传播
output = model(input_tensor)
print(output.shape)