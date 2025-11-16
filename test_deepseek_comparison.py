#!/usr/bin/env python3
"""
DeepSeek对比测试脚本
用于测试和验证DeepSeek对比脚本的功能
"""

import sys
import os

def test_imports():
    """测试必要的导入"""
    try:
        import requests
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from rouge_score import rouge_scorer
        from bert_score import score as bert_score
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        print("✅ 所有必要的包都已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少必要的包: {e}")
        print("请安装缺少的包:")
        print("pip install requests pandas matplotlib seaborn numpy rouge-score bert-score nltk")
        return False

def check_api_keys():
    """检查API密钥配置"""
    print("\n🔑 API密钥配置检查:")
    
    # 检查DeepSeek对比脚本是否存在
    if not os.path.exists("deepseek_comparison.py"):
        print("❌ deepseek_comparison.py 文件不存在")
        return False
    
    with open("deepseek_comparison.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "YOUR_MODEL_API_URL_HERE" in content:
        print("⚠️  您的模型API地址未配置")
    else:
        print("✅ 您的模型API地址已配置")
    
    if "YOUR_DEEPSEEK_API_KEY_HERE" in content:
        print("⚠️  DeepSeek API密钥未配置")
    else:
        print("✅ DeepSeek API密钥已配置")
    
    return True

def show_usage_instructions():
    """显示使用说明"""
    print("\n📖 使用说明:")
    print("=" * 50)
    print("1. 配置API信息:")
    print("   - 编辑 deepseek_comparison.py 文件")
    print("   - 将 YOUR_MODEL_API_URL_HERE 替换为您的模型API地址")
    print("   - 将 YOUR_DEEPSEEK_API_KEY_HERE 替换为您的DeepSeek API密钥")
    print()
    print("2. 运行对比评估:")
    print("   python deepseek_comparison.py")
    print()
    print("3. 自定义测试提示:")
    print("   - 编辑 deepseek_comparison.py 中的 test_prompts 列表")
    print("   - 添加您想要测试的提示")
    print()
    print("4. 查看结果:")
    print("   - deepseek_comparison_results.csv (详细结果)")
    print("   - deepseek_comparison_results.png (可视化图表)")
    print()
    print("5. 自定义评估指标:")
    print("   - 可以修改 calculate_* 方法来添加新的评估指标")

def show_api_example():
    """显示API调用示例"""
    print("\n🔧 API调用示例:")
    print("=" * 50)
    
    print("您的模型API调用示例:")
    print("""
import requests

url = "YOUR_MODEL_API_URL"
payload = {
    "prompt": "请写一首诗",
    "max_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    result = response.json()
    generated_text = result.get("text", result.get("generated_text", ""))
    print(generated_text)
""")
    
    print("\nDeepSeek API调用示例:")
    print("""
import requests

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Authorization": "Bearer YOUR_DEEPSEEK_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user",
            "content": "请写一首诗"
        }
    ],
    "max_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9
}

response = requests.post(url, json=payload, headers=headers)
if response.status_code == 200:
    result = response.json()
    generated_text = result["choices"][0]["message"]["content"]
    print(generated_text)
""")

def main():
    """主测试函数"""
    print("🚀 DeepSeek对比测试脚本")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        return
    
    # 检查API配置
    check_api_keys()
    
    # 显示使用说明
    show_usage_instructions()
    
    # 显示API示例
    show_api_example()
    
    print("\n✅ 测试完成！")
    print("请按照上述说明配置API信息后运行对比评估。")

if __name__ == "__main__":
    main()
