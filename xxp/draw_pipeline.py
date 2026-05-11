import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off') # 隐藏坐标轴

    # 定义颜色
    c_source = '#E1F5FE' # 浅蓝
    c_process = '#FFF3E0' # 浅橙
    c_verify = '#E8F5E9' # 浅绿
    c_final = '#F3E5F5' # 浅紫
    edge_color = '#333333'

    # 定义绘制盒子的函数
    def draw_box(x, y, w, h, text, title, color):
        # 阴影
        shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1", fc='#DDDDDD', ec='none', zorder=1)
        ax.add_patch(shadow)
        # 盒子
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec=edge_color, lw=1.5, zorder=2)
        ax.add_patch(box)
        # 标题
        ax.text(x + w/2, y + h - 0.3, title, ha='center', va='center', fontsize=9, fontweight='bold', color='#555555', zorder=3)
        # 内容
        ax.text(x + w/2, y + h/2 - 0.1, text, ha='center', va='center', fontsize=10, fontweight='normal', color='black', zorder=3)

    # 定义绘制箭头的函数
    def draw_arrow(x_start, y_start, x_end, y_end):
        ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='#555555'))

    # 1. Data Source Phase
    draw_box(0.5, 2.5, 2.0, 1.5, "GB Standards (PDF)\nJGJ Codes\nLocal Regulations", "Step 1: Data Source", c_source)
    
    draw_arrow(2.7, 3.25, 3.3, 3.25)

    # 2. Generation Phase (RAG)
    draw_box(3.5, 2.5, 2.5, 1.5, "Parsing & Chunking\nvector DB Retrieval\nLLM Generation", "Step 2: Semi-Auto Generation", c_process)

    draw_arrow(6.2, 3.25, 6.8, 3.25)

    # 3. Verification Phase
    draw_box(7.0, 2.0, 2.0, 2.5, "De-duplication\nDe-ambiguity\nExpert Review (>90%)", "Step 3: Human-in-the-Loop", c_verify)

    draw_arrow(9.2, 3.25, 9.8, 3.25)

    # 4. Final Output
    draw_box(10.0, 2.75, 1.5, 1.0, "10,144 Questions\n38 Domains", "Norma-MESBench", c_final)

    # 添加一些装饰性标注
    # RAG 标注
    ax.text(4.75, 1.8, "Prompt Engineering:\n- Role: Supervisor\n- Task: Violation Check", 
            ha='center', fontsize=8, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    
    # 专家标注
    ax.text(8.0, 1.3, "Registered Engineers\nTriple-Check Protocol", 
            ha='center', fontsize=8, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))

    plt.title('Figure 2: Data Construction Pipeline', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('pipeline_placeholder.pdf', format='pdf')
    print("Generated pipeline_placeholder.pdf")
    plt.close()

if __name__ == "__main__":
    draw_pipeline()