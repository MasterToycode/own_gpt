import os
import sentencepiece as spm

def train_tokenizer_with_symbols():
    input_file = "data_processed.txt"
    
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return
    
    print(f"使用数据文件: {input_file}")
    
    # 注意：自定义符号统一用 <END>，不要再用 [EOS]
    spm.SentencePieceTrainer.Train(
        f"--input={input_file} "
        f"--model_prefix=tokenizer "
        f"--vocab_size=8000 "
        f"--model_type=bpe "
        f"--character_coverage=1.0 "
        f"--user_defined_symbols=<END>,关键词:,诗词: "  # 添加自定义符号
        f"--unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3"
    )
    
    print("✅ 分词器训练完成，生成了 tokenizer.model 和 tokenizer.vocab 文件")
    print("✅ 已包含 <END>、关键词:、诗词: 符号")


def test_tokenizer():
    """
    测试分词器，验证 <END> 标记是否正确处理
    """
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer.model")
    
    # 测试包含 <END> 的文本
    test_text = "风，雾，寂寞：随风飘过 从没有找到真正的我 <END>"
    
    tokens = sp.encode(test_text, out_type=int)
    pieces = sp.encode(test_text, out_type=str)
    
    print("\n分词器测试结果:")
    print(f"原始文本: {test_text}")
    print(f"Token IDs: {tokens}")
    print(f"子词切分: {pieces}")
    
    # 检查是否包含 <END> 标记
    if "<END>" in pieces:
        eos_index = pieces.index("<END>")
        print(f"✅ <END> 标记位置: 第 {eos_index} 个 token")
    else:
        print("❌ <END> 标记未找到")
    
    # 解码测试
    decoded = sp.decode(tokens)
    print(f"解码回文本: {decoded}")

def main():
    print("开始训练包含 <END> 标记的分词器...")
    print("="*40)
    
    # 1. 训练分词器
    train_tokenizer_with_symbols()
    
    # 2. 测试分词器
    test_tokenizer()
    
    print("\n" + "="*40)
    print("分词器训练完成！")

if __name__ == "__main__":
    main()
