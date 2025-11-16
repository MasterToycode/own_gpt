# finetune_end.py
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
eval_iters = 200
d_model = 512
h = 8
Nx = 6
dropout_rate = 0.2
lr_rate = 1e-4
max_seq_len = 2048

# 停止标记增强参数
stop_token_weight = 1
enable_stop_training = True

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)
torch.manual_seed(1337)

# --------------------------- tokenizer ------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

def encode(s):
    return sp.encode(s, out_type=int)

def decode(tokens):
    # tokens can be list[int] or torch.tensor
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    text = sp.decode(tokens)
    if "<END>" in text:
        text = text.split("<END>")[0]
    return text.strip()

vocab_size = sp.get_piece_size()
print(f"词汇表大小: {vocab_size}")

# 获取特殊 id（END, PAD）
end_token_id = sp.piece_to_id("<END>")
if end_token_id == -1:
    raise ValueError("tokenizer 中没有 <END>，请用 user_defined_symbols 添加并重训 tokenizer")
# 自动探测 pad_id（常见训练时为1）
def detect_pad_id(sp):
    for cand in ["<pad>","<PAD>","<blank>","<PAD>","[PAD]"]:
        pid = sp.piece_to_id(cand)
        if pid != -1:
            return pid
    # fallback to 1
    return 1

pad_id = detect_pad_id(sp)
print(f"<END> id={end_token_id}, pad_id={pad_id}")

# --------------------------- 数据加载（按 sample grouping） ---------------------------
all_lines = []
data_path = "data.txt"   # 你的原始数据（每组有关键词/诗词/<END>，组之间有空行）
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found")

# 说明性提示，让模型知道任务目标不是简单拼接
instruction_prompt = "请根据给定的关键词，创作一首诗句。\n"

# <<< 关键修改：把若干行合并为一条 sample，直到遇到文本行为 "<END>"
with open(data_path, 'r', encoding='utf-8') as f:
    buf = []
    for raw in f:
        line = raw.rstrip("\n")
        if line.strip() == "":
            # 空行作为组间分隔（跳过）
            continue
        buf.append(line.strip())
        # 假定文件中每组以独立一行 "<END>" 结尾
        if line.strip() == "<END>":
            # 在样本前添加说明性提示
            sample_text = instruction_prompt + "\n".join(buf)  # 保持换行有助于 tokenizer 建模
            tokens = encode(sample_text)
            if not tokens:
                buf = []
                continue
            # 确保结尾为 end_token_id
            if tokens[-1] != end_token_id:
                tokens = tokens + [end_token_id]
            if len(tokens) <= max_seq_len:
                all_lines.append(tokens)
            buf = []
    # 如果文件末尾没有以 <END> 结尾的残余 buf，忽略或打印警告
    if buf:
        print("警告：文件末尾有未以 <END> 终止的残余样本（已忽略）：", buf[:2])

print(f"解析样本数: {len(all_lines)}")
if len(all_lines) == 0:
    raise RuntimeError("数据解析后样本数为0，请检查 data.txt 格式（每组必须以单独一行 <END> 结尾））")

split_90perc = int(0.9 * len(all_lines))
train_lines = all_lines[:split_90perc]
valid_lines = all_lines[split_90perc:]
print(f"训练样本数: {len(train_lines)}, 验证样本数: {len(valid_lines)}")

# --------------------------- 分析 <END> 分布（可选） ---------------------------
def analyze_stop_token_distribution(n=200):
    print("\n分析 <END> 分布：")
    sample_cnt = min(n, len(train_lines))
    end_positions = []
    found = 0
    for tokens in train_lines[:sample_cnt]:
        if end_token_id in tokens:
            pos = tokens.index(end_token_id)
            end_positions.append(pos)
            found += 1
    if found:
        print(f"前 {sample_cnt} 个样本中，包含 <END> 的数量: {found}/{sample_cnt}")
        print(f"<END> 平均位置: {np.mean(end_positions):.1f}")
    else:
        print("未发现 <END>，请检查 tokenizer 或样本是否包含文字 '<END>' 行。")
analyze_stop_token_distribution()

# --------------------------- 增强的损失函数（考虑 pad & end 加权） ---------------------------
class EnhancedLoss(nn.Module):
    def __init__(self, stop_token_id, stop_weight=2.0, pad_id=1):
        super().__init__()
        self.stop_token_id = stop_token_id
        self.stop_weight = stop_weight
        self.pad_id = pad_id

    def forward(self, logits, targets):
        """
        logits: (B, T, V)
        targets: (B, T)
        返回: total_loss (Tensor), standard_loss (float), stop_loss (float)
        """
        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)            # (B*T, V)
        targets_flat = targets.view(-1)             # (B*T,)

        # per-position loss (忽略 pad)
        loss_flat = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_id, reduction='none')  # (B*T,)
        mask = (targets_flat != self.pad_id).float()   # 有效位置

        # standard loss: 平均在非 pad 上
        denom = mask.sum().clamp(min=1.0)
        standard_loss = (loss_flat * mask).sum() / denom

        # 停止标记的 loss（只在带有 stop token 的位置上取 mean）
        stop_mask = (targets_flat == self.stop_token_id)
        stop_loss = torch.tensor(0.0, device=standard_loss.device)
        if enable_stop_training and stop_mask.any():
            stop_loss_vals = loss_flat[stop_mask]
            if stop_loss_vals.numel() > 0:
                stop_loss = stop_loss_vals.mean()
                total_loss = standard_loss + self.stop_weight * stop_loss
                return total_loss, standard_loss.item(), float(stop_loss.item())

        return standard_loss, float(standard_loss.item()), 0.0

# --------------------------- batch （按整条样本 pad） ---------------------------
def get_batch(split, batch_size_override=None):
    current_batch_size = batch_size_override if batch_size_override else batch_size
    dataset = train_lines if split == "train" else valid_lines
    # 随机采样（允许重复）
    idxs = np.random.randint(0, len(dataset), current_batch_size)
    batch_lines = [dataset[i] for i in idxs]

    x = [torch.tensor(l[:-1], dtype=torch.long) for l in batch_lines]  # input
    y = [torch.tensor(l[1:], dtype=torch.long) for l in batch_lines]   # target (预测下一个 token)
    max_len = max(len(xx) for xx in x)

    x = torch.stack([F.pad(xx, (0, max_len - len(xx)), value=pad_id) for xx in x]).to(device)
    y = torch.stack([F.pad(yy, (0, max_len - len(yy)), value=pad_id) for yy in y]).to(device)
    return x, y

# --------------------------- 验证函数 ---------------------------
@torch.no_grad()
def estimate_loss_and_ppl(model, criterion):
    result = {}
    model.eval()

    for split in ['train', 'valid']:
        std_losses = []
        stop_losses = []
        for e in range(eval_iters):
            X, Y = get_batch(split, batch_size_override=4)
            logits, _ = model(X, Y)   # logits (B, T, V)
            total_loss, std_loss, s_loss = criterion(logits, Y)
            std_losses.append(std_loss)
            stop_losses.append(s_loss)
            del X, Y, logits
            if device == 'cuda':
                torch.cuda.empty_cache()
        avg_std = float(np.mean(std_losses)) if len(std_losses) > 0 else float('nan')
        avg_stop = float(np.mean(stop_losses)) if len(stop_losses) > 0 else 0.0
        ppl = math.exp(avg_std) if not math.isinf(avg_std) else float('inf')
        result[f'{split}_loss'] = avg_std
        result[f'{split}_ppl'] = ppl
        result[f'{split}_stop_loss'] = avg_stop

    model.train()
    return result

# --------------------------- 保存模型 ---------------------------
def save_model(model, optimizer, iteration, train_losses, valid_losses, train_ppls, valid_ppls, final=False):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
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


# --------------------------- 加载已训练模型 ---------------------------
def load_pretrained_model(pretrained_path="saved_models/gpt_model_enhanced_stop_20251003_200243.pth"):
    model = MemoryOptimizedBigramLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        h=h,
        Nx=Nx,
        dropout_rate=dropout_rate
    ).to(device)

    if pretrained_path is None or not os.path.exists(pretrained_path):
        print("未指定或未找到预训练权重，will start from scratch.")
        return model, None

    try:
        # <<< 修正：去掉不存在的参数 weights_only
        checkpoint = torch.load(pretrained_path, map_location=device,weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # 过滤掉可能的非参数键（例如 mask buffers 等）
        filtered_state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ 成功加载已训练模型权重")
        return model, checkpoint
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("将从头开始训练...")
        return model, None

# --------------------------- 可选：测试 end-rate ---------------------------
def test_end_rate(model, prompts, top_k=50, temp=0.9, max_new=300):
    end_id = end_token_id
    pad = pad_id
    cnt = 0
    for p in prompts:
        ids = torch.tensor([encode(p)], dtype=torch.long, device=device)
        gen = model.generate(ids, max_new, temperature=temp, top_k=top_k, eos_token_id=end_id)
        gen_ids = gen[0].tolist()
        if end_id in gen_ids:
            cnt += 1
    print(f"end_rate: {cnt}/{len(prompts)} = {cnt/len(prompts):.3f}")

# --------------------------- 主训练 ---------------------------
def main(pretrained_path=None):
    model, pretrained_ckpt = load_pretrained_model(pretrained_path)
    # 清零 pad embedding（避免 pad 影响）
    with torch.no_grad():
        if pad_id < model.token_embedding.weight.shape[0]:
            model.token_embedding.weight[pad_id].zero_()

    criterion = EnhancedLoss(end_token_id, stop_token_weight, pad_id=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

    if pretrained_ckpt and pretrained_ckpt.get('optimizer_state_dict') is not None:
        try:
            optimizer.load_state_dict(pretrained_ckpt['optimizer_state_dict'])
            print("✅ 加载优化器状态")
        except Exception:
            print("⚠️ 无法加载优化器状态（忽略）")

    train_losses, valid_losses, train_ppls, valid_ppls = [], [], [], []

    print("开始增强停止标记训练...")
    try:
        for iter in range(num_iter):
            if iter % eval_interval == 0:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                results = estimate_loss_and_ppl(model, criterion)
                train_losses.append(results['train_loss'])
                valid_losses.append(results['valid_loss'])
                train_ppls.append(results['train_ppl'])
                valid_ppls.append(results['valid_ppl'])
                print(f"step {iter}: train_loss={results['train_loss']:.4f}, valid_loss={results['valid_loss']:.4f}, "
                      f"train_ppl={results['train_ppl']:.2f}, valid_ppl={results['valid_ppl']:.2f}")
                if results.get('train_stop_loss', 0.0) > 0:
                    print(f"         stop_loss={results['train_stop_loss']:.4f}")

            xb, yb = get_batch("train")
            logits, _ = model(xb, yb)
            loss, std_loss, stop_loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if iter % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n训练中断，保存当前进度...")
        save_model(model, optimizer, iter, train_losses, valid_losses, train_ppls, valid_ppls, final=False)
    except RuntimeError as e:
        print(f"\n运行时错误: {e}")
        save_model(model, optimizer, iter, train_losses, valid_losses, train_ppls, valid_ppls, final=False)
        raise e
    
    plot_training_curves(train_losses, valid_losses, train_ppls, valid_ppls)
    save_model(model, optimizer, num_iter, train_losses, valid_losses, train_ppls, valid_ppls, final=True)


    # --------------- 测试停止功能 ---------------
    print("\n测试停止功能（若要更严格请增大 max_new）:")
    test_prompts = [
        "请根据给定的关键词，创作一首诗句。\n关键词: 风 雾 寂寞\n诗词:",
        "请根据给定的关键词，创作一首诗句。\n关键词: 信 天涯 晚风\n诗词:",
        "请根据给定的关键词，创作一首诗句。\n关键词: 贴心 改变 自信\n诗词:"
    ]
    test_end_rate(model, test_prompts, top_k=50, temp=0.9, max_new=300)

if __name__ == "__main__":
    # 可选传入已训练 checkpoint 路径
    pretrained_checkpoint_path = "saved_models/gpt_model_enhanced_stop_20251003_200243.pth"
    main(pretrained_checkpoint_path)
