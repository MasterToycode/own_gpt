import torch
import sentencepiece as spm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 检查系统字体
import platform
if platform.system() == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
elif platform.system() == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Heiti SC']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']

class ModelComparisonEvaluator:
    def __init__(self):
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
        
        # 加载您的模型
        self.your_model = self.load_your_model()
        
        # 【修改点 1】: 替换为中文模型名称，保留您的模型
        self.model_names = [
            "Your Model",
            "GPT2 (中文)",
            "Dialogue GPT2 (中文)"
        ]

        # 新的模型 ID 映射
        self.hf_model_ids = {
            "GPT2 (中文)": "uer/gpt2-chinese-cluecorpussmall",
            "Dialogue GPT2 (中文)": "IDEA-CCNL/Wenzhong-GPT2-110M" # <-- 已修正
        }

        # 加载多个对比模型
        self.models = {
            'your_model': ('Your Model', self.your_model, None),
            'gpt2_chinese': ('GPT2 (中文)', *self.load_gpt2_chinese_model()),
            'dialogue_gpt2': ('Dialogue GPT2 (中文)', *self.load_dialogue_gpt2_model())
        }
        
        # 初始化评估器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
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
    
    def load_your_model(self):
        """加载您的GPT模型"""
        from model_optimized import MemoryOptimizedBigramLM
        
        model = MemoryOptimizedBigramLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            h=self.h,
            Nx=self.Nx,
            dropout_rate=self.dropout_rate
        )
        
        try:
            checkpoint = torch.load("saved_models/gpt_model_final_20251003_124248.pth", 
                                  map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'mask' not in k}
            model.load_state_dict(filtered_state_dict, strict=False)
            print("✅ 成功加载您的GPT模型")
        except Exception as e:
            print(f"❌ 加载您的模型失败: {e}")
            return None
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_distilgpt2_model(self):
        """加载DistilGPT2模型"""
        try:
            model_name = "distilgpt2"  # 82M参数，6层，768 hidden
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            model.eval()
            print("✅ 成功加载DistilGPT2模型 (82M参数)")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 加载DistilGPT2模型失败: {e}")
            return None, None
    
    def load_gpt2_model(self):
        """加载标准GPT2模型"""
        try:
            model_name = "gpt2"  # 124M参数
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            model.eval()
            print("✅ 成功加载GPT2模型 (124M参数)")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 加载GPT2模型失败: {e}")
            return None, None
    
    def load_tinystories_model(self):
        """加载TinyStories模型"""
        try:
            # 使用一个较小的模型作为TinyStories的替代
            model_name = "microsoft/DialoGPT-small"  # 约117M参数，作为小模型对比
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            model.eval()
            print("✅ 成功加载TinyStories替代模型 (117M参数)")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 加载TinyStories模型失败: {e}")
            return None, None
    
    def load_gpt2_chinese_model(self):
        """加载中文GPT2模型"""
        try:
            model_name = "uer/gpt2-chinese-cluecorpussmall"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            model.eval()
            print("✅ 成功加载中文GPT2模型")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 加载中文GPT2模型失败: {e}")
            return None, None
    
    def load_dialogue_gpt2_model(self):
        """加载对话GPT2模型"""
        try:
            model_name = "IDEA-CCNL/Wenzhong-GPT2-110M"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = model.to(self.device)
            model.eval()
            print("✅ 成功加载对话GPT2模型")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 加载对话GPT2模型失败: {e}")
            return None, None
    
    def generate_with_your_model(self, prompt, max_new_tokens=200):
        """使用您的模型生成文本"""
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
    
    def generate_with_model(self, model_name, prompt, max_new_tokens=200):
        """使用指定模型生成文本"""
        model_info = self.models.get(model_name)
        if not model_info or model_info[1] is None:
            return ""
        
        display_name, model, tokenizer = model_info
        
        if model_name == 'your_model':
            # 使用您的模型
            return self.generate_with_your_model(prompt, max_new_tokens)
        else:
            # 使用其他模型
            poetry_prompt = f"请根据以下关键词创作一首优美的诗歌：{prompt}\n诗歌："
            
            inputs = tokenizer.encode(poetry_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.3
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_start = generated_text.find(poetry_prompt) + len(poetry_prompt)
                response = generated_text[response_start:].strip()
                
                # 清理乱码：移除非中文字符和特殊符号
                import re
                cleaned_response = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef，。！？；：、\n\r]', '', response)
                cleaned_response = re.sub(r'[，。！？；：、]{2,}', '，', cleaned_response)
                cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
                
                if not cleaned_response:
                    return response
                
                return cleaned_response
    
    def calculate_bleu_score(self, generated, reference=None):
        """计算BLEU分数"""
        if reference is None:
            # 如果没有参考文本，使用prompt作为参考
            reference = [generated.split()[:5]]  # 使用前几个词作为参考
        
        smoothie = SmoothingFunction().method4
        try:
            score = sentence_bleu([reference], generated.split(), smoothing_function=smoothie)
            return score
        except:
            return 0.0
    
    def calculate_rouge_l(self, generated, reference=None):
        """计算ROUGE-L分数"""
        if reference is None:
            reference = generated[:50]  # 使用生成文本的前50个字符作为参考
        
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    
    def calculate_bertscore(self, generated, reference=None):
        """计算BERTScore"""
        if reference is None:
            reference = generated  # 自参考
        
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
        """计算连贯性分数（基于句子长度和结构）"""
        sentences = text.split('。')
        if len(sentences) < 2:
            return 0.5
        
        # 简单的连贯性评估：句子长度变化和多样性
        sentence_lengths = [len(sent) for sent in sentences if len(sent) > 0]
        if len(sentence_lengths) < 2:
            return 0.5
        
        # 句子长度标准差（适中的变化更好）
        length_std = np.std(sentence_lengths)
        coherence = 1.0 - min(length_std / 20, 1.0)  # 标准化
        
        return coherence
    
    def evaluate_single_prompt(self, prompt):
        """评估单个提示的所有模型输出"""
        metrics = {}
        
        for model_name, (display_name, model, tokenizer) in self.models.items():
            if model is None:
                continue
                
            output = self.generate_with_model(model_name, prompt)
            
            metrics[model_name] = {
                'display_name': display_name,
                'output': output,
                'bleu': self.calculate_bleu_score(output),
                'rouge_l': self.calculate_rouge_l(output),
                'bertscore': self.calculate_bertscore(output),
                'distinct_1': self.calculate_distinct_n(output, 1),
                'distinct_2': self.calculate_distinct_n(output, 2),
                'repetition_rate': self.calculate_repetition_rate(output),
                'coherence': self.calculate_coherence_score(output),
                'length': len(output)
            }
        
        return metrics
    
    def run_comparison(self):
        """运行完整的对比评估"""
        print("开始模型对比评估...")
        print("=" * 80)
        
        all_results = []
        
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\n进度: {i}/{len(self.test_prompts)}")
            result = self.evaluate_single_prompt(prompt)
            result['prompt'] = prompt
            all_results.append(result)
        
        return all_results
    
    def analyze_results(self, all_results):
        """分析并可视化结果"""
        # 提取所有模型的数据
        model_scores = {}
        for model_name in self.models.keys():
            model_scores[model_name] = []
        
        for result in all_results:
            for model_name, metrics in result.items():
                if model_name != 'prompt' and model_name in model_scores:
                    model_scores[model_name].append(metrics)
        
        # 创建DataFrame用于分析
        model_dfs = {}
        for model_name, scores in model_scores.items():
            if scores:
                model_dfs[model_name] = pd.DataFrame(scores)
        
        # 计算平均分数
        metrics = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 
                  'repetition_rate', 'coherence', 'length']
        
        avg_scores = {}
        for model_name, df in model_dfs.items():
            for metric in metrics:
                avg_scores[f'{model_name}_{metric}'] = df[metric].mean()
        
        # 打印结果摘要
        print("\n" + "="*80)
        print("多模型对比评估结果摘要")
        print("="*80)
        
        for metric in metrics:
            print(f"\n{metric.upper():<15}:")
            model_avgs = []
            for model_name in self.models.keys():
                if model_name in model_dfs:
                    avg = avg_scores[f'{model_name}_{metric}']
                    display_name = self.models[model_name][0]
                    model_avgs.append((display_name, avg))
                    print(f"  {display_name:<20}: {avg:.4f}")
            
            # 找出最佳模型
            if model_avgs:
                best_model = max(model_avgs, key=lambda x: x[1])
                print(f"  最佳模型: {best_model[0]} ({best_model[1]:.4f})")
        
        return model_dfs, avg_scores
    
    def create_visualizations(self, model_dfs, avg_scores):
        """创建可视化图表 - 展示3个模型的结果"""
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig1 = plt.figure(figsize=(18, 6))
        fig1.suptitle('Model Performance Comparison Analysis(picture 1/2 )', fontsize=16, fontweight='bold')
        
        # 定义模型颜色和标签
        model_colors = {
            'your_model': 'skyblue',
            'gpt2_chinese': 'lightcoral',
            'dialogue_gpt2': 'gold'
        }
        
        model_labels = {
            'your_model': 'Your Model',
            'gpt2_chinese': 'GPT2 (中文)',
            'dialogue_gpt2': 'Dialogue GPT2 (中文)'
        }
        
        # 第一排：3个主要对比图
        # 主要指标对比 - 柱状图
        ax1 = fig1.add_subplot(1, 3, 1) # 1行3列的第1个
        metrics_to_plot = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']
        metric_names = ['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence']
        
        x = np.arange(len(metrics_to_plot))
        width = 0.2
        
        for i, model_name in enumerate(self.models.keys()):
            if model_name in model_dfs:
                model_avgs = [avg_scores[f'{model_name}_{metric}'] for metric in metrics_to_plot]
                ax1.bar(x + i*width - width*1.5, model_avgs, width, 
                       label=model_labels[model_name], alpha=0.8, color=model_colors[model_name])
        
        ax1.set_xlabel('Evaluation Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Main Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names, rotation=45, fontsize=10)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 重复率对比 - 箱线图
        ax2 = fig1.add_subplot(1, 3, 2) # 1行3列的第2个
        repetition_data = []
        labels = []
        for model_name in self.models.keys():
            if model_name in model_dfs:
                repetition_data.append(model_dfs[model_name]['repetition_rate'])
                labels.append(model_labels[model_name])
        
        box_plot = ax2.boxplot(repetition_data, labels=labels, patch_artist=True)
        
        # 设置颜色
        for i, (patch, model_name) in enumerate(zip(box_plot['boxes'], self.models.keys())):
            if model_name in model_colors:
                patch.set_facecolor(model_colors[model_name])
        
        ax2.set_ylabel('Repetition Rate', fontsize=12)
        ax2.set_title('Repetition Rate Distribution', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 输出长度对比 - 箱线图
        ax3 = fig1.add_subplot(1, 3, 3) # 1行3列的第3个
        length_data = []
        labels = []
        for model_name in self.models.keys():
            if model_name in model_dfs:
                length_data.append(model_dfs[model_name]['length'])
                labels.append(model_labels[model_name])
        
        length_plot = ax3.boxplot(length_data, labels=labels, patch_artist=True)
        
        for i, (patch, model_name) in enumerate(zip(length_plot['boxes'], self.models.keys())):
            if model_name in model_colors:
                patch.set_facecolor(model_colors[model_name])
        
        ax3.set_ylabel('Output Length (characters)', fontsize=12)
        ax3.set_title('Output Length Comparison', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('model_comparison_results_1.png', dpi=300, bbox_inches='tight')


        fig2 = plt.figure(figsize=(18, 6))
        fig2.suptitle('Model Deep Analysis and Trends (picture 2/2)', fontsize=16, fontweight='bold')
        # 第二排：3个分析图
        # 指标相关性热力图 - 您的模型
        ax4 = fig2.add_subplot(1, 3, 1) # 1行3列的第1个
        if 'your_model' in model_dfs:
            correlation_matrix_your = model_dfs['your_model'][['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']].corr()
            sns.heatmap(correlation_matrix_your, annot=True, cmap='coolwarm', center=0, ax=ax4, 
                       xticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                       yticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                       annot_kws={"size": 9})
            ax4.set_title('Your Model: Metric Correlations', fontsize=14, fontweight='bold')
        
        # 指标相关性热力图 - 中文GPT2模型
        ax5 = fig2.add_subplot(1, 3, 2) # 1行3列的第1个
        if 'gpt2_chinese' in model_dfs:
            correlation_matrix_gpt2 = model_dfs['gpt2_chinese'][['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']].corr()
            sns.heatmap(correlation_matrix_gpt2, annot=True, cmap='coolwarm', center=0, ax=ax5,
                       xticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                       yticklabels=['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Distinct-2', 'Coherence'],
                       annot_kws={"size": 9})
            ax5.set_title('GPT2 (中文): Metric Correlations', fontsize=14, fontweight='bold')
        
        # 性能雷达图 - 所有模型
        ax6 = fig2.add_subplot(1, 3, 3) # 1行3列的第1个
        
        # 选择几个关键指标
        radar_metrics = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'coherence']
        radar_names = ['BLEU', 'ROUGE-L', 'BERTScore', 'Distinct-1', 'Coherence']
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for model_name in self.models.keys():
            if model_name in model_dfs:
                model_radar = [avg_scores[f'{model_name}_{metric}'] for metric in radar_metrics]
                
                # 归一化到0-1范围
                max_vals = [max([avg_scores[f'{m}_{metric}'] for m in self.models.keys() if m in model_dfs]) 
                           for metric in radar_metrics]
                model_radar_norm = [model_radar[i] / max_vals[i] if max_vals[i] > 0 else 0 
                                  for i in range(len(radar_metrics))]
                model_radar_norm += model_radar_norm[:1]
                
                ax6.plot(angles, model_radar_norm, 'o-', linewidth=2, 
                        label=model_labels[model_name], color=model_colors[model_name])
                ax6.fill(angles, model_radar_norm, alpha=0.25, color=model_colors[model_name])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(radar_names, fontsize=10)
        ax6.set_ylim(0, 1)
        ax6.set_title('Performance Radar Chart', fontsize=14, fontweight='bold')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('model_comparison_results_2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig1,fig2

def main():
    """主函数：运行完整的模型对比评估"""
    print("🚀 开始多模型对比评估")
    print("=" * 80)
    
    # 创建评估器
    evaluator = ModelComparisonEvaluator()
    
    # 运行对比评估
    all_results = evaluator.run_comparison()
    
    # 分析结果
    model_dfs, avg_scores = evaluator.analyze_results(all_results)
    
    # 创建可视化图表
    print("\n📊 正在生成可视化图表...")
    evaluator.create_visualizations(model_dfs, avg_scores)
    
    # 保存详细结果到CSV
    detailed_results = []
    for result in all_results:
        row = {'prompt': result['prompt']}
        for model_name, metrics in result.items():
            if model_name != 'prompt':
                display_name = metrics['display_name']
                row[f'{display_name}_output'] = metrics['output']
                row[f'{display_name}_bleu'] = metrics['bleu']
                row[f'{display_name}_rouge_l'] = metrics['rouge_l']
                row[f'{display_name}_bertscore'] = metrics['bertscore']
                row[f'{display_name}_distinct_1'] = metrics['distinct_1']
                row[f'{display_name}_distinct_2'] = metrics['distinct_2']
                row[f'{display_name}_repetition_rate'] = metrics['repetition_rate']
                row[f'{display_name}_coherence'] = metrics['coherence']
                row[f'{display_name}_length'] = metrics['length']
        detailed_results.append(row)
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('detailed_comparison_results.csv', index=False, encoding='utf-8-sig')
    
    print("\n✅ 评估完成！")
    print("📁 生成的文件:")
    print("   - detailed_comparison_results.csv (详细结果)")
    print("   - model_comparison_results.png (可视化图表)")
    
    # 最终总结
    print("\n🎯 最终总结:")
    model_wins = {}
    for model_name in evaluator.models.keys():
        if model_name in model_dfs:
            model_wins[model_name] = 0
    
    metrics = ['bleu', 'rouge_l', 'bertscore', 'distinct_1', 'distinct_2', 'coherence']
    
    for metric in metrics:
        best_score = -1
        best_models = []
        for model_name in evaluator.models.keys():
            if model_name in model_dfs:
                score = avg_scores[f'{model_name}_{metric}']
                if score > best_score:
                    best_score = score
                    best_models = [model_name]
                elif score == best_score:
                    best_models.append(model_name)
        
        for model_name in best_models:
            model_wins[model_name] += 1
    
    print("各模型获胜指标数:")
    for model_name, wins in model_wins.items():
        display_name = evaluator.models[model_name][0]
        print(f"  {display_name}: {wins} 个指标")
    
    # 找出总体最佳模型
    best_model = max(model_wins.items(), key=lambda x: x[1])
    best_display_name = evaluator.models[best_model[0]][0]
    print(f"\n🏆 总体最佳模型: {best_display_name} (在 {best_model[1]} 个指标上表现最佳)")
    
    if best_model[0] == 'your_model':
        print("🎉 恭喜！您的模型在多数指标上表现最佳！")
    else:
        print(f"⚠️ {best_display_name} 在多数指标上表现更好，您的模型仍有改进空间")

if __name__ == "__main__":
    main()
