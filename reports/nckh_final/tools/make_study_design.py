"""Draw the two controlled comparisons, without depicting unmeasured effects."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
fig, ax = plt.subplots(figsize=(10.4,5.0))
ax.set_xlim(0,10.4)
ax.set_ylim(0,5)
ax.axis('off')


def box(x,y,w,h,text,color='#eef2f4',edge='#52606a',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.06',
                              linewidth=.8, facecolor=color, edgecolor=edge))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.5)


def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
                                 linewidth=1,color='#52606a'))

ax.text(.05,4.75,'V8  •  Giá trị dự báo trong định tuyến',fontsize=12,fontweight='bold',color='#195d60')
box(.05,3.6,2.05,.8,'Quan sát hiện tại\nvà lịch sử')
box(2.65,3.36,2.55,1.3,'EWMA hiện tại\nGNN dự báo 5 s\nMLP dự báo 5 s',color='#e8f3ef')
box(5.75,3.6,2.0,.8,'Cùng luật\nghim tuyến ETX')
box(8.3,3.6,2.05,.8,'PDR vòng kín\nRelay đứng yên')
arrow(2.14,4,2.6,4)
arrow(5.24,4,5.7,4)
arrow(7.79,4,8.25,4)
ax.text(5.2,3.0,'Đối chứng hệ thống: OLSR-ETX-fast; không dùng để thay endpoint chính.',
        ha='center',fontsize=9,color='#404b52')

ax.text(.05,2.45,'V9  •  Giá trị dự báo khi kích hoạt chuyển động',fontsize=12,fontweight='bold',color='#195d60')
box(.05,1.13,2.05,.8,'Cùng định tuyến\ntheo số đo hiện tại')
box(2.65,.89,2.55,1.3,'Trigger phản ứng\nTrigger GNN\nTrigger MLP',color='#e8f3ef')
box(5.75,1.13,2.0,.8,'Cùng luật\nbố trí relay')
box(8.3,1.13,2.05,.8,'Giây mất kết nối\nPDR và quãng đường',size=9)
arrow(2.14,1.53,2.6,1.53)
arrow(5.24,1.53,5.7,1.53)
arrow(7.79,1.53,8.25,1.53)
ax.text(5.2,.5,'Đối chứng bổ sung: relay đứng yên. Trọng số dự báo giữ nguyên từ v8.',
        ha='center',fontsize=9,color='#404b52')
ax.text(5.2,.08,'GNN/MLP học có giám sát; luật hành động cố định. Mọi điều khiển chạy trong ns-3.',
        ha='center',fontsize=9,color='#404b52')
fig.tight_layout(pad=.5)
for extension in ('pdf','png'):
    fig.savefig(ROOT / f'images/fig_v8_v9_design.{extension}',dpi=220,bbox_inches='tight')
print('Wrote v8/v9 design figures')
