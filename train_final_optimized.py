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
from model_optimized import MemoryOptimizedBigramLM  # 使用内存优化的模型

# --------------------------- 超参数 ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
num_iter = 10000
eval_interval = 500
eval_iters = 200
d_model = 512
h = 8
Nx = 6
dropout_rate = 0.2
lr_rate = 1e-3
max_seq_len = 2048  # 进一步增大最大序列长度以处理所有文本

# 内存优化参数
valid_batch_size = 8  # 验证时使用更小的batch size
enable_mixed_precision = True  # 混合精度训练

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

print(f"过滤后训练样本数: {len(train_lines)}, 验证样本数: {len(valid_lines)}")

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

# --------------------------- 内存优化的验证函数 ---------------------------
@torch.no_grad()
def estimate_loss_and_ppl(model):
    result = {}
    model.eval()
    
    for split in ['train', 'valid']:
        losses = []
        for e in range(eval_iters):
            X, Y = get_batch(split, batch_size_override=valid_batch_size)
            
            if enable_mixed_precision:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(X, Y)
            else:
                logits, loss = model(X, Y)
                
            losses.append(loss.item())
            
            # 显式清理GPU内存
            del X, Y, logits, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        avg_loss = np.mean(losses)
        ppl = math.exp(avg_loss)
        result[f'{split}_loss'] = avg_loss
        result[f'{split}_ppl'] = ppl
    
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
        filename = f"{model_save_dir}/gpt_model_final_{timestamp}.pth"
    else:
        filename = f"{model_save_dir}/gpt_model_checkpoint_{timestamp}_iter_{iteration}.pth"
    torch.save(checkpoint, filename)
    print(f"模型已保存到: {filename}")

    info_filename = f"{model_save_dir}/training_info_{timestamp}.json"
    with open(info_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'iteration': iteration,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_ppls': train_ppls,
            'valid_ppls': valid_ppls
        }, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存到: {info_filename}")

# --------------------------- 绘图 ---------------------------
def plot_training_curves(train_losses, valid_losses, train_ppls, valid_ppls):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    iterations = list(range(0, len(train_losses) * eval_interval, eval_interval))
    ax1.plot(iterations, train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(iterations, valid_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(iterations, train_ppls, label='Train PPL', color='blue', linewidth=2)
    ax2.plot(iterations, valid_ppls, label='Validation PPL', color='red', linewidth=2)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Perplexity (PPL)')
    ax2.set_title('Validation PPL')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_save_dir}/training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------- 内存监控函数 ---------------------------
def print_memory_usage(step):
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Step {step}: GPU内存 - 已分配: {allocated:.2f}GB, 保留: {reserved:.2f}GB")

# --------------------------- 主训练 ---------------------------
def main():
    # 设置内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model = MemoryOptimizedBigramLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        h=h,
        Nx=Nx,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)
    
    if enable_mixed_precision:
        scaler = torch.amp.GradScaler('cuda')  # 半精度梯度缩放
    else:
        scaler = None
        
    accum_steps = 1  # 梯度累积步数，可根据显存调节
    train_losses, valid_losses, train_ppls, valid_ppls = [], [], [], []

    print("开始训练...")
    print(f"设备: {device}, 训练样本数: {len(train_lines)}, 验证样本数: {len(valid_lines)}")
    print(f"内存优化设置: 验证batch_size={valid_batch_size}, 混合精度={enable_mixed_precision}")

    try:
        for iter in range(num_iter):
            if iter % eval_interval == 0:
                # 验证前清理内存
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                results = estimate_loss_and_ppl(model)
                train_losses.append(results['train_loss'])
                valid_losses.append(results['valid_loss'])
                train_ppls.append(results['train_ppl'])
                valid_ppls.append(results['valid_ppl'])
                print(f"step {iter}: train_loss={results['train_loss']:.4f}, "
                      f"valid_loss={results['valid_loss']:.4f}, "
                      f"train_ppl={results['train_ppl']:.2f}, valid_ppl={results['valid_ppl']:.2f}")
                

            optimizer.zero_grad(set_to_none=True)

            # 梯度累积循环
            for _ in range(accum_steps):
                xb, yb = get_batch("train")
                
                if enable_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        logits, loss = model(xb, yb)
                        loss = loss / accum_steps  # 梯度累积缩放
                    scaler.scale(loss).backward()
                else:
                    logits, loss = model(xb, yb)
                    loss = loss / accum_steps  # 梯度累积缩放
                    loss.backward()
                
                # 清理训练batch的内存
                del xb, yb, logits, loss

            if enable_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
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

    plot_training_curves(train_losses, valid_losses, train_ppls, valid_ppls)
    save_model(model, optimizer, num_iter, train_losses, valid_losses, train_ppls, valid_ppls, final=True)

    # --------------------------- 生成示例 ---------------------------
    print("\n生成示例文本:")
    prompt = "关键词: 风 雾 寂寞:"
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=200)[0].tolist()
    print(decode(generated_tokens))
    print(f"\n训练完成，模型与曲线已保存到 '{model_save_dir}'")

    
if __name__ == "__main__":
    main()
