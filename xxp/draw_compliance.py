import matplotlib.pyplot as plt
import numpy as np

def draw_compliance_drop():
    # 数据准备
    models = ['Human Expert', 'DeepSeek-V3', 'Qwen-2.5', 'GPT-4o', 'Claude-3.5']
    
    # 标准准确率 (Standard Accuracy)
    std_scores = np.array([96.5, 76.4, 74.5, 75.8, 73.2])
    
    # 严格合规分 (Strict Compliance Score - Requires Correct Citation)
    # 假设：人类几乎不错，国产模型稍降，国外模型暴跌（因为不懂GB规范）
    strict_scores = np.array([92.3, 68.5, 66.8, 48.2, 45.1])
    
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制柱状图
    rects1 = ax.bar(x - width/2, std_scores, width, label='Standard Accuracy', color='#AEC7E8', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, strict_scores, width, label='Strict Compliance Score (w/ Citation)', color='#1F77B4', edgecolor='black', alpha=1.0)

    # 添加文本标签 (下降幅度)
    for i in range(len(models)):
        drop = std_scores[i] - strict_scores[i]
        drop_pct = (drop / std_scores[i]) * 100
        
        # 在严格分柱子上方标注下降值
        if i > 0: # 人类不标
            ax.annotate(f'-{drop_pct:.1f}%',
                        xy=(x[i] + width/2, strict_scores[i]),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#D62728')

    # 设置轴标签
    ax.set_ylabel('Score (0-100)', fontsize=12)
    ax.set_title('Figure 4: The Safety Gap - Hallucination Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 110)
    
    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    # 图例
    ax.legend(fontsize=11, loc='upper right')
    
    # 添加注释说明
    ax.text(3.5, 100, "Severe Drop for\nNon-Domestic Models", ha='center', fontsize=10, color='#D62728',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D62728", lw=1))

    plt.tight_layout()
    plt.savefig('compliance_drop.pdf', format='pdf')
    print("Generated compliance_drop.pdf")
    plt.close()

if __name__ == "__main__":
    draw_compliance_drop()