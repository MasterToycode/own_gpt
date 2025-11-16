import torch
import sentencepiece as spm
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# 导入OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("❌ 未安装OpenAI SDK，请运行: pip install openai")
    OPENAI_AVAILABLE = False

# 设置英文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.size'] = 12

class DeepSeekLocalComparison:
    def __init__(self, deepseek_api_key):
        """
        初始化比较器
        
        Args:
            deepseek_api_key: DeepSeek API密钥
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load("tokenizer.model")
        self.vocab_size = self.sp.get_piece_size()
        print(f"词汇表大小: {self.vocab_size}")
        
        # 模型参数
        self.d_model = 512
        self.max_seq_len = 2048
        self.h = 8
        self.Nx = 6
        self.dropout_rate = 0.2
        
        # 加载您的本地模型
        self.your_model = self.load_your_local_model()
        
        self.deepseek_api_key = deepseek_api_key
        
        # 测试提示
        self.test_prompts = [
            "关键词: 信 天涯 晚风",
            "关键词: 风 雾 寂寞", 
            "关键词: 贴心 改变 自信",
            "关键词: 午夜 寒冬 心动",
            "关键词: 思考 推理 分析",
            "关键词: 月光 思念 远方",
            "关键词: 梦想 坚持 成功",
            "关键词: 春天 希望 新生",
            "关键词: 学习 进步 成长",
            "关键词: 友谊 信任 陪伴"
        ]
        
        # 初始化评估器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        print("🚀 DeepSeek对比评估器初始化完成")
        print(f"您的模型: 本地训练模型")
        print(f"DeepSeek模型: API调用")
        print(f"测试提示数量: {len(self.test_prompts)}")
    
    def load_your_local_model(self):
        """加载您的本地GPT模型"""
        try:
            from model_optimized import MemoryOptimizedBigramLM
            
            model = MemoryOptimizedBigramLM(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                max_seq_len=self.max_seq_len,
                h=self.h,
                Nx=self.Nx,
                dropout_rate=self.dropout_rate
            )
            
            # 尝试加载最新的模型
            checkpoint_paths = [
                "saved_models/gpt_model_enhanced_stop_20251005_192151.pth"
            ]
            
            loaded = False
            for checkpoint_path in checkpoint_paths:
                try:
                    checkpoint = torch.load(checkpoint_path, 
                                          map_location=self.device, weights_only=False)
                    state_dict = checkpoint['model_state_dict']
                    filtered_state_dict = {k: v for k, v in state_dict.items() if 'mask' not in k}
                    model.load_state_dict(filtered_state_dict, strict=False)
                    print(f"✅ 成功加载您的GPT模型: {checkpoint_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"❌ 加载 {checkpoint_path} 失败: {e}")
                    continue
            
            if not loaded:
                print("❌ 所有模型文件加载失败")
                return None
            
        except Exception as e:
            print(f"❌ 加载您的模型失败: {e}")
            return None
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def generate_with_your_model(self, prompt, max_new_tokens=200):
        """使用您的本地模型生成文本"""
        if self.your_model is None:
            return ""
        
        temperature = 0.8
        top_k = 50
        repetition_penalty = 1.3
        
        prompt_tokens = self.sp.encode(prompt, out_type=int)
        context = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            generated_tokens = self.your_model.generate(
                context, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )[0].tolist()
            
            generated_text = self.sp.decode(generated_tokens)
            response_start = generated_text.find(prompt) + len(prompt)
            response = generated_text[response_start:].strip()
            
            return response
    
    def call_deepseek_api(self, prompt, max_tokens=200):
        """调用DeepSeek API - 使用官方OpenAI SDK"""
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI SDK未安装，请运行: pip install openai")
            return ""
            
        try:
            # 为DeepSeek对话模型添加明确的指令
            enhanced_prompt = self.enhance_prompt_for_deepseek(prompt)
            
            print(f"🔍 正在调用DeepSeek API (使用OpenAI SDK)...")
            print(f"   原始提示: {prompt[:50]}...")
            print(f"   增强提示: {enhanced_prompt[:80]}...")
            
            # 按照官方示例创建客户端
            client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            
            # 调用API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的诗歌创作助手，擅长创作优美的中文诗歌。"},
                    {"role": "user", "content": enhanced_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                stream=False
            )
            
            content = response.choices[0].message.content
            print(f"✅ DeepSeek API调用成功，输出长度: {len(content)}")
            return content
            
        except Exception as e:
            print(f"❌ 调用DeepSeek API时出错: {e}")
            return ""
    
    def enhance_prompt_for_deepseek(self, prompt):
        """为DeepSeek对话模型增强提示词"""
        # 根据提示类型添加不同的指令
        if "关键词:" in prompt:
            # 提取关键词
            keywords = prompt.replace("关键词:", "").strip()
            enhanced_prompt = f"""请根据以下关键词创作一首优美的中文诗歌：

关键词：{keywords}

要求：
1. 必须是一首完整的诗歌
2. 诗歌要有意境和美感
3. 合理运用给定的关键词
4. 诗歌格式可以是现代诗或古体诗
5. 直接输出诗歌内容，不要添加其他说明

请开始创作："""
        elif "请写一首关于" in prompt:
            # 提取主题
            theme = prompt.replace("请写一首关于", "").replace("的诗", "").strip()
            enhanced_prompt = f"""请创作一首关于{theme}的优美中文诗歌。

要求：
1. 必须是一首完整的诗歌
2. 围绕{theme}主题展开
3. 诗歌要有意境和美感
4. 直接输出诗歌内容，不要添加其他说明

请开始创作："""
        elif "描述" in prompt or "解释" in prompt or "写一段" in prompt:
            # 说明文类型
            enhanced_prompt = f"""{prompt}

请用优美、流畅的中文进行回答，直接给出内容，不要添加其他说明。"""
        else:
            # 其他类型提示
            enhanced_prompt = f"""{prompt}

请用优美、流畅的中文进行回答，直接给出内容，不要添加其他说明。"""
        
        return enhanced_prompt
    
    def calculate_bleu_score(self, generated, reference=None):
        """计算BLEU分数"""
        if reference is None:
            reference = [generated.split()[:5]]
        
        smoothie = SmoothingFunction().method4
        try:
            score = sentence_bleu([reference], generated.split(), smoothing_function=smoothie)
            return score
        except:
            return 0.0
    
    def calculate_rouge_l(self, generated, reference=None):
        """计算ROUGE-L分数"""
        if reference is None:
            reference = generated[:50]
        
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    def calculate_bertscore(self, generated, reference=None):
        """计算BERTScore"""
        if reference is None:
            reference = generated
        
        try:
            P, R, F1 = bert_score([generated], [reference], lang="zh", verbose=False)
            return F1.item()
        except:
            return 0.0
    
    def calculate_distinct_n(self, text, n):
        """计算distinct-n指标"""
        words = text.split()
        if len(words) < n:
            return 0.0
        
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def calculate_repetition_rate(self, text):
        """计算重复率"""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        repeated_count = 0
        total_pairs = len(words) - 1
        
        for i in range(total_pairs):
            if words[i] == words[i+1]:
                repeated_count += 1
        
        return repeated_count / total_pairs if total_pairs > 0 else 0.0
    
    def calculate_coherence_score(self, text):
        """计算连贯性分数"""
        sentences = text.split('。')
        if len(sentences) < 2:
            return 0.5
        
        sentence_lengths = [len(sent) for sent in sentences if len(sent) > 0]
        if len(sentence_lengths) < 2:
            return 0.5
        
        length_std = np.std(sentence_lengths)
        coherence = 1.0 - min(length_std / 20, 1.0)
        
        return coherence
    
    def evaluate_single_prompt(self, prompt):
        """评估单个提示的两个模型输出"""
        print(f"处理提示: {prompt}")
        
        # 调用两个模型
        your_model_output = self.generate_with_your_model(prompt)
        time.sleep(1)  # 避免API限流
        deepseek_output = self.call_deepseek_api(prompt)
        
        print(f"您的模型输出长度: {len(your_model_output)}")
        print(f"DeepSeek输出长度: {len(deepseek_output)}")
        
        # 计算指标
        metrics = {
            'your_model': {
                'output': your_model_output,
                'bleu': self.calculate_bleu_score(your_model_output),
                'rouge_l': self.calculate_rouge_l(your_model_output),
                'bertscore': self.calculate_bertscore(your_model_output),
                'distinct_1': self.calculate_distinct_n(your_model_output, 1),
                'distinct_2': self.calculate_distinct_n(your_model_output, 2),
                'repetition_rate': self.calculate_repetition_rate(your_model_output),
                'coherence': self.calculate_coherence_score(your_model_output),
                'length': len(your_model_output)
            },
            'deepseek': {
                'output': deepseek_output,
                'bleu': self.calculate_bleu_score(deepseek_output),
                'rouge_l': self.calculate_rouge_l(deepseek_output),
                'bertscore': self.calculate_bertscore(deepseek_output),
                'distinct_1': self.calculate_distinct_n(deepseek_output, 1),
                'distinct_2': self.calculate_distinct_n(deepseek_output, 2),
                'repetition_rate': self.calculate_repetition_rate(deepseek_output),
                'coherence': self.calculate_coherence_score(deepseek_output),
                'length': len(deepseek_output)
            }
        }
        
        return metrics
    
    def run_comparison(self):
        """运行完整的对比评估"""
        print("开始DeepSeek对比评估...")
        print("=" * 80)
        
        all_results = []
        
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\n进度: {i}/{len(self.test_prompts)}")
            result = self.evaluate_single_prompt(prompt)
            result['prompt'] = prompt
            all_results.append(result)
            
            # 每3个提示后休息一下，避免API限流
            if i % 3 == 0:
                print("休息5秒...")
                time.sleep(5)
        
        return all_results
    
    def analyze_results(self, all_results):
        """分析并可视化结果"""
        # 提取数据
        your_model_scores = []
        deepseek_scores = []
        
        for result in all_results:
            your_model_scores.append(result['your_model'])
            deepseek_scores.append(result['deepseek'])
        
        # 创建DataFrame
        your_model_df = pd.DataFrame(your_model_scores)
        deepseek_df = pd.DataFrame(deepseek_scores)
        
        # 计算平均分数
        metrics = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 
                  'repetition_rate', 'coherence', 'length']
        
        your_model_avg = {metric: your_model_df[metric].mean() for metric in metrics}
        deepseek_avg = {metric: deepseek_df[metric].mean() for metric in metrics}
        
        # 打印结果摘要
        print("\n" + "="*80)
        print("DeepSeek对比评估结果摘要")
        print("="*80)
        
        for metric in metrics:
            print(f"\n{metric.upper():<15}:")
            print(f"  您的模型: {your_model_avg[metric]:.4f}")
            print(f"  DeepSeek: {deepseek_avg[metric]:.4f}")
            
            if your_model_avg[metric] > deepseek_avg[metric]:
                print(f"  🎉 您的模型领先: +{your_model_avg[metric] - deepseek_avg[metric]:.4f}")
            elif deepseek_avg[metric] > your_model_avg[metric]:
                print(f"  ⚠️ DeepSeek领先: +{deepseek_avg[metric] - your_model_avg[metric]:.4f}")
            else:
                print(f"  🤝 平局")
        
        return your_model_df, deepseek_df, your_model_avg, deepseek_avg
    
    def create_visualizations(self, your_model_df, deepseek_df, your_model_avg, deepseek_avg):
        """Create visualization charts"""
        # Set chart style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DeepSeek vs Your Local Model - Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Color settings
        colors = ['#3498db', '#e74c3c']  # Blue - Your model, Red - DeepSeek
        
        # 1. Main metrics comparison - Bar chart
        ax1 = axes[0, 0]
        metrics_to_plot = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']
        metric_names = ['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence']
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        your_model_values = [your_model_avg[metric] for metric in metrics_to_plot]
        deepseek_values = [deepseek_avg[metric] for metric in metrics_to_plot]
        
        ax1.bar(x - width/2, your_model_values, width, label='Your Model', color=colors[0], alpha=0.8)
        ax1.bar(x + width/2, deepseek_values, width, label='DeepSeek', color=colors[1], alpha=0.8)
        
        ax1.set_xlabel('Evaluation Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Main Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names, rotation=45, fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Repetition rate comparison - Box plot
        ax2 = axes[0, 1]
        repetition_data = [your_model_df['repetition_rate'], deepseek_df['repetition_rate']]
        box_plot = ax2.boxplot(repetition_data, labels=['Your Model', 'DeepSeek'], patch_artist=True)
        
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i])
        
        ax2.set_ylabel('Repetition Rate', fontsize=12)
        ax2.set_title('Repetition Rate Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Output length comparison - Box plot
        ax3 = axes[0, 2]
        length_data = [your_model_df['length'], deepseek_df['length']]
        length_plot = ax3.boxplot(length_data, labels=['Your Model', 'DeepSeek'], patch_artist=True)
        
        for i, patch in enumerate(length_plot['boxes']):
            patch.set_facecolor(colors[i])
        
        ax3.set_ylabel('Output Length (characters)', fontsize=12)
        ax3.set_title('Output Length Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance radar chart
        ax4 = axes[1, 0]
        radar_metrics = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'coherence']
        radar_names = ['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Coherence']
        
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        your_model_radar = [your_model_avg[metric] for metric in radar_metrics]
        deepseek_radar = [deepseek_avg[metric] for metric in radar_metrics]
        
        # Normalization
        max_vals = [max(your_model_radar[i], deepseek_radar[i]) for i in range(len(radar_metrics))]
        your_model_radar_norm = [your_model_radar[i] / max_vals[i] if max_vals[i] > 0 else 0 
                               for i in range(len(radar_metrics))]
        deepseek_radar_norm = [deepseek_radar[i] / max_vals[i] if max_vals[i] > 0 else 0 
                             for i in range(len(radar_metrics))]
        
        your_model_radar_norm += your_model_radar_norm[:1]
        deepseek_radar_norm += deepseek_radar_norm[:1]
        
        ax4.plot(angles, your_model_radar_norm, 'o-', linewidth=2, label='Your Model', color=colors[0])
        ax4.fill(angles, your_model_radar_norm, alpha=0.25, color=colors[0])
        ax4.plot(angles, deepseek_radar_norm, 'o-', linewidth=2, label='DeepSeek', color=colors[1])
        ax4.fill(angles, deepseek_radar_norm, alpha=0.25, color=colors[1])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(radar_names, fontsize=10)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Radar Chart', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Metric correlation heatmap - Your model
        ax5 = axes[1, 1]
        correlation_matrix_your = your_model_df[['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']].corr()
        sns.heatmap(correlation_matrix_your, annot=True, cmap='coolwarm', center=0, ax=ax5, 
                   xticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                   yticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                   annot_kws={"size": 9})
        ax5.set_title('Your Model: Metric Correlation', fontsize=14, fontweight='bold')
        
        # 6. Metric correlation heatmap - DeepSeek
        ax6 = axes[1, 2]
        correlation_matrix_deepseek = deepseek_df[['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']].corr()
        sns.heatmap(correlation_matrix_deepseek, annot=True, cmap='coolwarm', center=0, ax=ax6,
                   xticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                   yticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                   annot_kws={"size": 9})
        ax6.set_title('DeepSeek: Metric Correlation', fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('deepseek_local_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_detailed_results(self, all_results):
        """保存详细结果到CSV"""
        detailed_results = []
        for result in all_results:
            row = {'prompt': result['prompt']}
            row['your_model_output'] = result['your_model']['output']
            row['deepseek_output'] = result['deepseek']['output']
            
            for metric in ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'repetition_rate', 'coherence', 'length']:
                row[f'your_model_{metric}'] = result['your_model'][metric]
                row[f'deepseek_{metric}'] = result['deepseek'][metric]
            
            detailed_results.append(row)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('deepseek_local_comparison_results.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 详细结果已保存到: deepseek_local_comparison_results.csv")
        
        return detailed_df

def main():
    """主函数：运行DeepSeek对比评估"""
    print("🚀 DeepSeek对比评估 - 本地模型 vs DeepSeek API")
    print("=" * 80)
    
    
    # 配置DeepSeek API密钥
    DEEPSEEK_API_KEY = "sk-e1c7fb08748f4f4fa642065595069962"  # 请替换为您的DeepSeek API密钥
    
    # 创建比较器
    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        print("❌ 请先配置您的DeepSeek API密钥")
        print("请编辑 deepseek_local_comparison.py 文件，将 YOUR_DEEPSEEK_API_KEY_HERE 替换为您的API密钥")
        return
    
    comparator = DeepSeekLocalComparison(deepseek_api_key=DEEPSEEK_API_KEY)
    
    # 检查模型是否加载成功
    if comparator.your_model is None:
        print("❌ 您的本地模型加载失败，无法进行对比")
        return
    
    # 运行对比评估
    all_results = comparator.run_comparison()
    
    # 分析结果
    your_model_df, deepseek_df, your_model_avg, deepseek_avg = comparator.analyze_results(all_results)
    
    # 创建可视化图表
    print("\n📊 正在生成可视化图表...")
    comparator.create_visualizations(your_model_df, deepseek_df, your_model_avg, deepseek_avg)
    
    # 保存详细结果
    comparator.save_detailed_results(all_results)
    
    # 最终总结
    print("\n🎯 最终总结:")
    metrics = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']
    
    your_model_wins = 0
    deepseek_wins = 0
    ties = 0
    
    for metric in metrics:
        your_score = your_model_avg[metric]
        deepseek_score = deepseek_avg[metric]
        
        if your_score > deepseek_score:
            your_model_wins += 1
        elif deepseek_score > your_score:
            deepseek_wins += 1
        else:
            ties += 1
    
    print(f"您的模型获胜指标数: {your_model_wins}")
    print(f"DeepSeek获胜指标数: {deepseek_wins}")
    print(f"平局指标数: {ties}")
    
    if your_model_wins > deepseek_wins:
        print(f"\n🏆 总体最佳模型: 您的本地模型 (在 {your_model_wins} 个指标上表现最佳)")
        print("🎉 恭喜！您的本地模型在多数指标上表现优于DeepSeek！")
    elif deepseek_wins > your_model_wins:
        print(f"\n🏆 总体最佳模型: DeepSeek (在 {deepseek_wins} 个指标上表现最佳)")
        print("⚠️ DeepSeek在多数指标上表现更好，您的模型仍有改进空间")
    else:
        print(f"\n🤝 总体平局 (双方各在 {your_model_wins} 个指标上表现最佳)")
        print("您的本地模型与DeepSeek表现相当！")
    
    print("\n✅ 评估完成！")
    print("📁 生成的文件:")
    print("   - deepseek_local_comparison_results.csv (详细结果)")
    print("   - deepseek_local_comparison_results.png (可视化图表)")

if __name__ == "__main__":
    main()
