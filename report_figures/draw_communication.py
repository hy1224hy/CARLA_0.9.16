import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(15,7.2)); fig.patch.set_facecolor('white'); ax.set(xlim=(0,15),ylim=(0,7.2)); ax.axis('off')
blue='#315d82'; orange='#a96530'; ink='#263746'
def box(x,y,w,h,text,color=blue,fill='#f0f5fa',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.07',linewidth=1.3,edgecolor=color,facecolor=fill,zorder=3))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=ink,zorder=4)
def arrow(points,color=ink,dashed=False):
 for a,b in zip(points[:-2],points[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.25,ls='--' if dashed else '-',zorder=2)
 ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,lw=1.25,color=color,linestyle='--' if dashed else '-',zorder=2))
for y,idx,c,fill in [(4.65,0,blue,'#f0f5fa'),(1.05,1,orange,'#fcf5ef')]:
 ax.add_patch(FancyBboxPatch((.15,y-.45),14.65,(2.3 if idx==0 else 1.9),boxstyle='round,pad=0.02,rounding_size=.13',edgecolor='#c6cdd4',facecolor='#fcfcfd',lw=1,zorder=0))
 ax.text(.4,y+(1.5 if idx==0 else 1.25),f'智能体 {idx}',fontweight='bold',color=c,fontsize=13)
 box(.5,y,1.35,.75,f'本地观测\n$o_{idx}$',c,fill)
 box(2.45,y,2.05,.75,f'独立消息编码器\n$E_{idx}$：MLP + tanh',c,fill)
 box(5.1,y,1.2,.75,f'发送消息\n$m_{idx}$',c,fill)
 box(7.8,y,2.25,.75,f'特征拼接\n$[o_{idx},\; \\widetilde m_{1-idx},\; b_{idx}]$',c,fill)
 box(10.65,y,1.65,.75,f'独立策略网络\nActor {idx}',c,fill)
 box(12.9,y,1.45,.75,f'动作分布\n$\\pi_{idx}(a_{idx}|\\cdot)$',c,fill)
 arrow([(1.85,y+.375),(2.45,y+.375)],c);arrow([(4.5,y+.375),(5.1,y+.375)],c)
 arrow([(10.05,y+.375),(10.65,y+.375)],c);arrow([(12.3,y+.375),(12.9,y+.375)],c)
 arrow([(1.2,y+.75),(1.2,y+1.1),(8.9,y+1.1),(8.9,y+.75)],c)
box(6.7,3.48,2.3,.65,'接收门控（智能体 0）\n$\\widetilde m_1=b_0m_1$',blue,'#f0f5fa',10)
box(3.65,2.62,2.3,.65,'接收门控（智能体 1）\n$\\widetilde m_0=b_1m_0$',orange,'#fcf5ef',10)
arrow([(5.7,4.65),(5.7,3.85),(4.8,3.85),(4.8,3.27)],blue)
arrow([(5.7,1.8),(5.7,2.15),(7.85,2.15),(7.85,3.48)],orange)
arrow([(9,3.8),(9.4,3.8),(9.4,4.65)],blue)
arrow([(5.95,2.94),(8.95,2.94),(8.95,1.8)],orange)
box(.55,2.65,2.25,.9,'有效通信掩码 $b_i$\n链路状态 × 通信使能',ink,'#f4f4f4',10)
arrow([(2.8,3.1),(3.65,3.1)],ink,True)
arrow([(1.65,3.55),(1.65,4.03),(6.7,4.03)],ink,True)
ax.text(10.25,3.15,'消息：$m_i=\\tanh(E_i(o_i))$\n策略输入维度：$d_o+d_m+1$\n编码器与 Actor 均不共享参数',fontsize=11,ha='left',va='center',color=ink,linespacing=1.9)
ax.text(7.5,.2,'实线：观测与消息流    虚线：通信控制    |    无通信时：接收消息与掩码均置零',ha='center',fontsize=10,color='#536170')
fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
p=Path('report_figures')
for ext in ['svg','pdf','png']: fig.savefig(p/f'communication_architecture.{ext}',dpi=300,facecolor='white')

