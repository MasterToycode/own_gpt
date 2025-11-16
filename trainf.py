import os
import json
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import matplotlib.pyplot as plt
from model_optimized import MemoryOptimizedBigramLM

# --------------------------- 超参数 ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
num_iter = 10000 
eval_interval = 500
eval_iters = 500
d_model = 512
h = 8
Nx = 6
dropout_rate = 0.2
lr_rate = 1e-4
max_seq_len = 2048

# 停止标记增强参数
stop_token_weight = 1  # 增加停止标记的损失权重
enable_stop_training = True  # 启用停止标记专门训练

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)
torch.manual_seed(1337)

# --------------------------- tokenizer ------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

def encode(s):
    return sp.encode(s, out_type=int)

def decode(tokens):
    text = sp.decode(tokens)
    if "<END>" in text:
        text = text.split("<END>")[0]
    return text.strip()

vocab_size = sp.get_piece_size()
print(f"词汇表大小: {vocab_size}")

# 获取<END>标记ID
end_token_id = sp.piece_to_id("<END>")
print(f"<END>标记ID: {end_token_id}")

# --------------------------- 数据加载 ------------------------
all_lines = []
with open('data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tokens = encode(line)
        # 过滤掉超过最大序列长度的序列
        if len(tokens) <= max_seq_len:
            all_lines.append(tokens)

split_90perc = int(0.9 * len(all_lines))
train_lines = all_lines[:split_90perc]
valid_lines = all_lines[split_90perc:]

print(f"训练样本数: {len(train_lines)}, 验证样本数: {len(valid_lines)}")

# 分析<END>标记在训练数据中的分布
def analyze_stop_token_distribution():
    """分析停止标记在训练数据中的分布"""
    print(f"\n分析<END>标记分布:")
    
    end_positions = []
    for tokens in train_lines[:100]:  # 分析前100个样本
        if end_token_id in tokens:
            pos = tokens.index(end_token_id)
            end_positions.append(pos)
            # 检查<END>是否在末尾
            if pos != len(tokens) - 1:
                print(f"警告: <END>不在末尾，位置: {pos}/{len(tokens)}")
    
    if end_positions:
        avg_position = np.mean(end_positions)
        print(f"<END>平均位置: {avg_position:.1f} (总长度)")
        print(f"包含<END>的样本比例: {len(end_positions)}/100")
    else:
        print("未找到<END>标记")

analyze_stop_token_distribution()

# --------------------------- 增强的损失函数 ---------------------------
class EnhancedLoss(nn.Module):
    def __init__(self, stop_token_id, stop_weight=2.0):
        super().__init__()
        self.stop_token_id = stop_token_id
        self.stop_weight = stop_weight
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        # 标准交叉熵损失
        standard_loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if enable_stop_training:
            # 增强停止标记的损失权重
            batch_size, seq_len, vocab_size = logits.shape
            
            # 找到目标中<END>标记的位置
            stop_mask = (targets == self.stop_token_id)
            
            if stop_mask.any():
                # 计算停止标记的损失
                stop_logits = logits[stop_mask]
                stop_targets = targets[stop_mask]
                stop_loss = self.criterion(stop_logits, stop_targets)
                
                # 加权组合损失
                total_loss = standard_loss + self.stop_weight * stop_loss
                return total_loss, standard_loss.item(), stop_loss.item()
        
        return standard_loss, standard_loss.item(), 0.0

# --------------------------- batch ---------------------------
def get_batch(split, batch_size_override=None):
    current_batch_size = batch_size_override if batch_size_override else batch_size
    dataset = train_lines if split == "train" else valid_lines
    batch_lines = [dataset[i] for i in np.random.randint(0, len(dataset), current_batch_size)]

    x = [torch.tensor(line[:-1], dtype=torch.long) for line in batch_lines]
    y = [torch.tensor(line[1:], dtype=torch.long) for line in batch_lines]

    max_len = max(len(xx) for xx in x)
    # Use padding token ID 1 instead of 0
    x = torch.stack([F.pad(xx, (0, max_len - len(xx)), value=1) for xx in x]).to(device)
    y = torch.stack([F.pad(yy, (0, max_len - len(yy)), value=1) for yy in y]).to(device)

    return x, y

# --------------------------- 验证函数 ---------------------------
@torch.no_grad()
def estimate_loss_and_ppl(model, criterion):
    result = {}
    model.eval()
    
    for split in ['train', 'valid']:
        losses = []
        stop_losses = []
        for e in range(eval_iters):
            X, Y = get_batch(split, batch_size_override=4)#验证batch减半
            
            logits, _ = model(X, Y)
            total_loss, standard_loss, stop_loss = criterion(logits, Y)
            
            losses.append(standard_loss)
            stop_losses.append(stop_loss)
            
            # 显式清理GPU内存
            del X, Y, logits
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_loss = np.mean(losses)
        avg_stop_loss = np.mean(stop_losses) if stop_losses[0] > 0 else 0.0
        ppl = math.exp(avg_loss)
        result[f'{split}_loss'] = avg_loss
        result[f'{split}_ppl'] = ppl
        result[f'{split}_stop_loss'] = avg_stop_loss
    
    model.train()
    return result

# --------------------------- 保存模型 ---------------------------
def save_model(model, optimizer, iteration, train_losses, valid_losses, train_ppls, valid_ppls, final=False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_ppls': train_ppls,
        'valid_ppls': valid_ppls,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'h': h,
        'Nx': Nx,
        'dropout_rate': dropout_rate,
        'save_time': timestamp
    }
    if final:
        filename = f"{model_save_dir}/gpt_model_enhanced_stop_{timestamp}.pth"
    else:
        filename = f"{model_save_dir}/gpt_model_checkpoint_enhanced_stop_{timestamp}_iter_{iteration}.pth"
    torch.save(checkpoint, filename)
    print(f"模型已保存到: {filename}")

# --------------------------- 加载已训练模型 ---------------------------
def load_pretrained_model():
    """加载已训练好的模型"""
    model = MemoryOptimizedBigramLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        h=h,
        Nx=Nx,
        dropout_rate=dropout_rate
    ).to(device)
    
    # 加载最新的训练模型权重
    try:
        checkpoint = torch.load("saved_models/gpt_model_enhanced_stop_20251003_200243.pth", map_location=device, weights_only=False)
        
        # 过滤掉mask相关的键，因为它们不是模型参数而是缓冲区
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'mask' not in k}
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ 成功加载已训练模型权重")
        print(f"已训练迭代次数: {checkpoint['iteration']}")
        print(f"最终训练损失: {checkpoint['train_losses'][-1]:.4f}")
        print(f"最终验证损失: {checkpoint['valid_losses'][-1]:.4f}")
        
        return model, checkpoint
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("将从头开始训练...")
        return model, None

# --------------------------- 主训练 ---------------------------
def main():
    # 加载已训练模型
    model, pretrained_checkpoint = load_pretrained_model()
    
    # 使用增强的损失函数
    criterion = EnhancedLoss(end_token_id, stop_token_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    
    # 如果加载了预训练模型，可以继续使用之前的优化器状态
    if pretrained_checkpoint and 'optimizer_state_dict' in pretrained_checkpoint:
        optimizer.load_state_dict(pretrained_checkpoint['optimizer_state_dict'])
        print("✅ 加载优化器状态")
    
    train_losses, valid_losses, train_ppls, valid_ppls = [], [], [], []
    train_stop_losses = []

    print("开始增强停止标记训练...")
    print(f"停止标记权重: {stop_token_weight}")
    print(f"启用停止训练: {enable_stop_training}")

    try:
        for iter in range(num_iter):
            if iter % eval_interval == 0:
                # 验证前清理内存
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                results = estimate_loss_and_ppl(model, criterion)
                train_losses.append(results['train_loss'])
                valid_losses.append(results['valid_loss'])
                train_ppls.append(results['train_ppl'])
                valid_ppls.append(results['valid_ppl'])
                train_stop_losses.append(results['train_stop_loss'])
                
                print(f"step {iter}: train_loss={results['train_loss']:.4f}, "
                      f"valid_loss={results['valid_loss']:.4f}, "
                      f"train_ppl={results['train_ppl']:.2f}, valid_ppl={results['valid_ppl']:.2f}")
                if results['train_stop_loss'] > 0:
                    print(f"         stop_loss={results['train_stop_loss']:.4f}")

            optimizer.zero_grad(set_to_none=True)

            xb, yb = get_batch("train")
            logits, _ = model(xb, yb)
            loss, standard_loss, stop_loss = criterion(logits, yb)
            
            loss.backward()
            optimizer.step()
            
            # 每100步清理一次GPU缓存
            if iter % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n训练中断，保存当前进度...")
        save_model(model, optimizer, iter, train_losses, valid_losses, train_ppls, valid_ppls, final=False)
    except torch.OutOfMemoryError as e:
        print(f"\n内存不足错误: {e}")
        print("尝试保存当前进度...")
        save_model(model, optimizer, iter, train_losses, valid_losses, train_ppls, valid_ppls, final=False)
        raise e

    save_model(model, optimizer, num_iter, train_losses, valid_losses, train_ppls, valid_ppls, final=True)

    # --------------------------- 测试停止功能 ---------------------------
    print("\n测试停止功能:")
    test_prompts = [
        "关键词: 风 雾 寂寞",
        "关键词: 信 天涯 晚风",
        "关键词: 贴心 改变 自信"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"测试: {prompt}")
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                context, 
                max_new_tokens=300,
                temperature=0.9,
                top_k=50,
                repetition_penalty=1.3,
                eos_token_id=end_token_id
            )[0].tolist()
            
            generated_text = sp.decode(generated_tokens)
            has_end = "<END>" in generated_text
            
            if has_end:
                end_pos = generated_text.find("<END>")
                response = generated_text[:end_pos].strip()
                print(f"✅ 成功使用<END>停止")
                print(f"输出: {response}")
            else:
                print(f"❌ 未使用<END>停止")
                print(f"输出: {generated_text}")

    print(f"\n增强停止标记训练完成，模型已保存到 '{model_save_dir}'")

if __name__ == "__main__":
    main()
