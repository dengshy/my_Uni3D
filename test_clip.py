from open_clip import create_model_and_transforms

# 指定模型和预训练权重
model_name = "RN50"  # 模型名称
pretrained_weights = "openai"  # 预训练权重名称

# 加载模型
model, preprocess, tokenizer = create_model_and_transforms(model_name, pretrained=pretrained_weights)

# 打印模型文本编码维度
text_embedding_dim = model.text_projection.shape[1]
print(f"Text embedding dimension for {model_name}: {text_embedding_dim}")
