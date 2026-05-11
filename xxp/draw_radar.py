import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    """
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, **closed):
            return super().fill(closed=True, *args, **closed)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return Circle((0.5, 0.5), 0.5)

    register_projection(RadarAxes)
    return theta

def draw_radar():
    # 数据定义 (基于您文档的推测数据)
    labels = ['Generalist', 'Specialist', 'Regulations\n(Evidence)', 'Case Analysis\n(Reasoning)', 'Practical']
    
    # 分数数据 (0-100)
    # Human: 全能，但在计算速度上可能稍弱，但这里指正确率，所以很高
    data = [
        ['Human Expert', [95, 98, 96, 94, 99]],
        ['DeepSeek-V3',  [78, 72, 85, 70, 65]], # 国产模型：法规强
        ['GPT-4o',       [76, 68, 60, 75, 72]]  # 国外模型：法规弱(幻觉)，但在推理/计算上不错
    ]
    
    N = len(labels)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    # 配色方案
    colors = ['#FFD700', '#D62728', '#1F77B4'] # 金色(专家), 红色(DeepSeek), 蓝色(GPT4)
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']

    for i, (title, case_data) in enumerate(data):
        ax.plot(theta, case_data, color=colors[i], linestyle=line_styles[i], marker=markers[i], linewidth=2, label=title)
        ax.fill(theta, case_data, facecolor=colors[i], alpha=0.15)

    ax.set_varlabels(labels)
    
    # 设置刻度
    ax.set_rgrids([20, 40, 60, 80, 100], labels=['20', '40', '60', '80', ''], angle=0, fontsize=9, color='grey')
    ax.set_ylim(0, 100)

    # 添加图例
    legend = ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize=10)
    
    # 标题
    plt.title('Figure 3: Capability Radar Chart\n(The Expert Gap)', y=1.08, fontsize=14, weight='bold')

    # 保存
    plt.savefig('radar_chart.pdf', format='pdf', bbox_inches='tight')
    print("Generated radar_chart.pdf")
    plt.close()

if __name__ == "__main__":
    draw_radar()