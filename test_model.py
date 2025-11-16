import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
from model import BigramLM

def load_model(model_path):
    """加载已保存的模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型实例
    model = BigramLM(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        block_size=checkpoint['block_size'],
        h=checkpoint['h'],
        Nx=checkpoint['Nx'],
        dropout_rate=checkpoint['dropout_rate']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 获取训练历史
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']
    train_ppls = checkpoint['train_ppls']
    valid_ppls = checkpoint['valid_ppls']
    iteration = checkpoint['iteration']
    
    return model, train_losses, valid_losses, train_ppls, valid_ppls, iteration

def plot_training_history(train_losses, valid_losses, train_ppls, valid_ppls, iteration):
    """绘制训练历史曲线"""
    eval_interval = 500  # 假设评估间隔为500
    iterations = list(range(0, len(train_losses) * eval_interval, eval_interval))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制loss曲线
    ax1.plot(iterations, train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(iterations, valid_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training History (Iteration: {iteration})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制PPL曲线
    ax2.plot(iterations, train_ppls, label='Train PPL', color='blue', linewidth=2)
    ax2.plot(iterations, valid_ppls, label='Validation PPL', color='red', linewidth=2)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Perplexity (PPL)')
    ax2.set_title(f'Perplexity History (Iteration: {iteration})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_text(model, sp, prompt="", max_new_tokens=200, device='cpu'):
    """使用模型生成文本"""
    model.eval()
    model.to(device)
    
    # 编码提示文本
    if prompt:
        encoded_prompt = sp.encode(prompt, out_type=int)
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # 生成文本
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
        generated_text = sp.decode(generated_tokens)
    
    return generated_text

def compare_models(model_paths, sp, device='cpu'):
    """比较多个模型的性能"""
    print("模型比较结果:")
    print("-" * 80)
    
    for model_path in model_paths:
        try:
            model, train_losses, valid_losses, train_ppls, valid_ppls, iteration = load_model(model_path)
            
            print(f"\n模型: {model_path}")
            print(f"训练迭代次数: {iteration}")
            print(f"最终训练损失: {train_losses[-1]:.4f}")
            print(f"最终验证损失: {valid_losses[-1]:.4f}")
            print(f"最终训练PPL: {train_ppls[-1]:.2f}")
            print(f"最终验证PPL: {valid_ppls[-1]:.2f}")
            
            # 生成示例文本
            generated_text = generate_text(model, sp, max_new_tokens=100, device=device)
            print(f"生成文本示例: {generated_text[:100]}...")
            
        except Exception as e:
            print(f"加载模型 {model_path} 失败: {e}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("lyric_tokenizer.model")
    
    # 列出可用的模型
    import os
    model_files = [f for f in os.listdir("saved_models") if f.endswith(".pth")]
    
    if not model_files:
        print("没有找到保存的模型文件")
        return
    
    # 按时间排序（最新的在前面）
    model_files.sort(reverse=True)
    
    print("可用的模型文件（按时间倒序）:")
    for i, model_file in enumerate(model_files):
        # 显示更友好的文件名
        if "final" in model_file:
            file_type = "最终模型"
        elif "checkpoint" in model_file:
            file_type = "检查点"
        else:
            file_type = "模型"
        
        # 提取时间信息
        import re
        time_match = re.search(r'(\d{8}_\d{6})', model_file)
        time_str = time_match.group(1) if time_match else "未知时间"
        
        print(f"{i+1}. {file_type} - {time_str} - {model_file}")
    
    # 选择要测试的模型
    try:
        choice = int(input("\n请选择要测试的模型编号: ")) - 1
        selected_model = os.path.join("saved_models", model_files[choice])
    except (ValueError, IndexError):
        print("无效的选择，使用第一个模型")
        selected_model = os.path.join("saved_models", model_files[0])
    
    # 加载模型
    model, train_losses, valid_losses, train_ppls, valid_ppls, iteration = load_model(selected_model)
    model.to(device)
    
    # 显示训练历史
    print(f"\n加载模型: {selected_model}")
    print(f"训练迭代次数: {iteration}")
    if train_losses:  # 如果有训练历史数据
        plot_training_history(train_losses, valid_losses, train_ppls, valid_ppls, iteration)
    else:
        print("该模型没有训练历史数据")
    
    # 交互式文本生成
    while True:
        print("\n" + "="*50)
        prompt = input("请输入提示文本（直接回车使用空提示，输入'quit'退出）: ")
        
        if prompt.lower() == 'quit':
            break
        
        max_tokens = input("请输入要生成的token数量（默认200）: ")
        try:
            max_tokens = int(max_tokens) if max_tokens else 200
        except ValueError:
            max_tokens = 200
        
        generated_text = generate_text(model, sp, prompt, max_tokens, device)
        print(f"\n生成的文本:")
        print(generated_text)

if __name__ == "__main__":
    main()
