"""Compare the three empirical advice tables requested on 2026-10-29.

python tools/visualize_three_advice.py
Read-only source audit; no simulation or model calls. See requirements-visualization.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_recent_results import setup_font, INK, MUTED, clean
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
WRITE_ROOT=Path(str(ROOT).lower())
if not (WRITE_ROOT.is_dir() and WRITE_ROOT.samefile(ROOT)):WRITE_ROOT=ROOT
REPORT=WRITE_ROOT/"reports/advice_deepseek_gpt_qwen_20261029"
MODELS=["DeepSeek","GPT","Qwen"]
COLORS={"DeepSeek":"#D27A2A","GPT":"#287AB5","Qwen":"#009E91"}
REPEATS={"DeepSeek":1,"GPT":2,"Qwen":2}
SEQ=LinearSegmentedColormap.from_list("advice_frequency",["#F1F5F7","#A1D7CB","#17958C","#17394F"])
PAIRS=[("DeepSeek","GPT"),("DeepSeek","Qwen"),("GPT","Qwen")]


def page(title,subtitle,number,foot):
    f=plt.figure(figsize=(16,9))
    f.text(.055,.958,"DEEPSEEK / GPT / QWEN  ·  EMPIRICAL ADVICE TABLES",fontsize=10,color=MUTED)
    f.text(.055,.901,title,fontsize=25,weight="bold")
    f.text(.055,.853,subtitle,fontsize=12,color=MUTED)
    f.add_artist(plt.Line2D([.055,.945],[.825,.825],transform=f.transFigure,color="#DCE5EB",lw=1))
    f.text(.055,.103,foot,fontsize=10.5,color=MUTED,va="top",linespacing=1.55)
    f.text(.055,.025,"来源：用户指定的三批日志 · 每表512局面 · 每局面DeepSeek 1答，GPT/Qwen各2答",fontsize=10,color=MUTED)
    f.text(.945,.025,number,ha="right",fontsize=11,color=MUTED)
    return f


def overview(frames):
    f=page("DeepSeek总体合作建议居中，但更倾向保持本人原行动",
           "全部512种局面等权；比例来自已保存的回复，未按仿真实际访问局面加权", "01",
           "保持原行动包括C→C和D→D，不能把它等同于促进合作。三批均为完整局面枚举。\nDeepSeek每局面只有1答，不能把其0/1观测当成模型的确定性选择，也不能由总体占比预测SPD稳态。")
    gs=f.add_gridspec(1,2,left=.09,right=.95,bottom=.23,top=.73,wspace=.25)
    for j,title in enumerate(["建议合作C的比例","保持本人当前行动的比例"]):
        ax=f.add_subplot(gs[0,j]);values=[]
        for name in MODELS:
            d=frames[name]
            values.append(d.is_c.mean() if j==0 else d.recommendation.eq(d.center).mean())
        ax.bar(MODELS,values,color=[COLORS[n] for n in MODELS],width=.55)
        for i,y in enumerate(values):
            ax.text(i,y+.025,f"{y:.2%}",ha="center",fontsize=17,weight="bold",color=COLORS[MODELS[i]])
            if j==0:
                d=frames[MODELS[i]]
                ax.text(i,y/2,f"{int(d.is_c.sum())}/{len(d)}",ha="center",va="center",color="white",fontsize=13)
        ax.set(ylim=(0,1.03),title=title,ylabel="记录中的建议比例")
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));clean(ax)
    return f


def transitions(frames):
    f=page("DeepSeek的主要特点：保留已有合作，很少将D转为C",
           "行是本人当前行动，列是建议行动；每行加总100%，小字为对应回复数 / 该行总回复数", "02",
           "本人原C时，DeepSeek 189/256条回复建议继续C；本人原D时，仅5/256条建议转为C。\n每行对全部256种对应局面等权；GPT/Qwen每局面两答，因此其行分母为512。不是实际演化转移率。")
    gs=f.add_gridspec(1,3,left=.075,right=.90,bottom=.26,top=.70,wspace=.24)
    for j,name in enumerate(MODELS):
        ax=f.add_subplot(gs[0,j]);d=frames[name]
        counts=np.array([[sum((d.center==s)&(d.recommendation==a)) for a in ["C","D"]] for s in ["C","D"]])
        den=counts.sum(1);p=counts/den[:,None]
        im=ax.imshow(p,cmap=SEQ,vmin=0,vmax=1,aspect="auto")
        for (i,k),v in np.ndenumerate(p):
            color="white" if v>.66 else INK
            ax.text(k,i-.08,f"{v:.2%}",ha="center",va="center",fontsize=21,weight="bold",color=color)
            ax.text(k,i+.19,f"{counts[i,k]}/{den[i]}",ha="center",va="center",fontsize=12,color=color)
        ax.set_xticks([0,1],["建议C","建议D"]);ax.set_yticks([0,1],["本人C","本人D"])
        ax.set_title(f"{name} · 每局面{REPEATS[name]}答",fontsize=16,pad=16)
        ax.set_xticks([.5],minor=True);ax.set_yticks([.5],minor=True);ax.grid(which="minor",color="white",lw=2)
        ax.tick_params(which="both",length=0,pad=9)
        for spine in ax.spines.values():spine.set_visible(False)
    f.colorbar(im,cax=f.add_axes([.93,.26,.014,.44]),format=PercentFormatter(1))
    return f


def neighbors(frames):
    f=page("同样的合作邻域，对原C与原D的建议很不一样",
           "按本人当前行动分面，再按四个直接邻居中的合作人数汇总；仿真建议表仍匹配完整九格", "03",
           "每个面板内，邻居C数0–4对应局面数依次16、64、96、64、16；每局面回复数为1/2/2。\n本人D且四邻全C：DeepSeek建议C为0/16，GPT为5/32，Qwen为29/32；差异限于本次记录。")
    gs=f.add_gridspec(1,2,left=.08,right=.95,bottom=.23,top=.72,wspace=.24)
    for j,s in enumerate(["C","D"]):
        ax=f.add_subplot(gs[0,j])
        for name,marker in zip(MODELS,["o","^","s"]):
            p=frames[name].loc[frames[name].center==s].groupby("cardinal_c").is_c.mean().reindex(range(5))
            ax.plot(range(5),p,color=COLORS[name],lw=2.5,marker=marker,ms=7,label=name)
        ax.set(xlim=(-.1,4.1),ylim=(-.04,1.08),xticks=range(5),xlabel="四个直接邻居中的合作人数",ylabel="建议合作C的比例",title=f"本人当前为{s}")
        ax.yaxis.set_major_formatter(PercentFormatter(1));clean(ax)
    f.legend(*f.axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.51,.82),ncol=3,frameon=False,fontsize=14)
    return f


def table_distance(states):
    f=page("DeepSeek更像哪张表？答案取决于本人原来的行动",
           "逐局面计算两表经验C比例的绝对差，再对局面平均；距离越小，本次记录越接近", "04",
           "DeepSeek单局面比例只能取0或1，GPT/Qwen可取0、0.5、1；混合回答保留0.5，不任意投票破平局。\n该距离描述本次经验表，不是模型总体相似度；重复数不同，距离也会受回答抽样影响。")
    ax=f.add_axes([.09,.24,.86,.48]);x=np.arange(3)
    for j,(subset,label,color) in enumerate([(states,"全部512局面",INK),(states[states.center=="C"],"本人C · 256局面",COLORS["Qwen"]),(states[states.center=="D"],"本人D · 256局面",COLORS["DeepSeek"])]):
        values=[(subset[a]-subset[b]).abs().mean()*100 for a,b in PAIRS]
        pos=x+(j-1)*.24;ax.bar(pos,values,width=.22,label=label,color=color)
        for xp,v in zip(pos,values):ax.text(xp,v+.8,f"{v:.2f}",ha="center",fontsize=12,color=color)
    ax.set(xticks=x,xticklabels=[f"{a} – {b}" for a,b in PAIRS],ylim=(0,55),ylabel="经验C比例平均绝对差（百分点）")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.20),ncol=3,frameon=False,fontsize=13);clean(ax)
    return f


def cases(frames):
    f=page("四个实际局面：相同建议占比可以隐藏不同选择",
           "中心橙框为本人；C为青绿、D为浅灰；每张图下方直接列出各次回复", "05",
           "示例1、2对照D→C与C→C；示例3、4保持本人及四直接邻居不变，只镜像移动一个D角。\n后两例DeepSeek回答不同，但每局面仅1答，不能区分稳定的位置偏好与抽样波动。示例不代表总体频率。")
    keys=["CCCCDCCCC","CDCDCDCDC","CDCDCDCDD","CDCDCDDDC"]
    captions=["本人D，四邻C，四角C","本人C，四邻D，四角C","本人C，四邻D，三角C","本人C，四邻D，三角C"]
    for j,(key,caption) in enumerate(zip(keys,captions)):
        x=.065+j*.232
        ax=f.add_axes([x+.018,.43,.17,.30]);a=np.array([c=="C" for c in key]).reshape(3,3)
        ax.imshow(a,cmap=ListedColormap(["#E8EDF2","#008F87"]),vmin=0,vmax=1,interpolation="nearest")
        for i in range(3):
            for k in range(3):ax.text(k,i,key[3*i+k],ha="center",va="center",fontsize=22,color="white" if a[i,k] else INK)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which="minor",color="white",lw=2);ax.tick_params(which="both",length=0)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        ax.add_patch(Rectangle((.5,.5),1,1,fill=False,edgecolor=COLORS["DeepSeek"],lw=3))
        ax.set_title(f"示例{j+1}",fontsize=15,pad=13)
        f.text(x+.103,.397,caption,ha="center",fontsize=11.5)
        for i,name in enumerate(MODELS):
            answers=frames[name].loc[frames[name].grid_key==key].sort_values("repetition").recommendation.tolist()
            f.text(x+.01,.337-i*.053,f"{name}：{' / '.join(answers)}",fontsize=14,color=COLORS[name])
    return f


def diagonals(states):
    f=page("对角信息也可能影响建议：同时控制直接与对角合作数",
           "列为模型，行为本人当前C/D；每个小格是该条件下的经验合作建议百分比", "06",
           "每格含1–36种局面，数量由直接及对角C数的组合决定；DeepSeek每局面1答，另两表各2答。\n这是枚举条件下的描述；同一计数组仍含不同位置排列，不能据此确定稳定的几何偏好或宏观因果效应。")
    gs=f.add_gridspec(2,3,left=.065,right=.91,bottom=.19,top=.77,wspace=.23,hspace=.49)
    for i,s in enumerate(["C","D"]):
        for j,name in enumerate(MODELS):
            ax=f.add_subplot(gs[i,j]);d=states[states.center==s]
            a=d.groupby(["diagonal_c","cardinal_c"])[name].mean().unstack().reindex(index=range(5),columns=range(5)).to_numpy()*100
            im=ax.imshow(a,cmap=SEQ,vmin=0,vmax=100,aspect="auto",origin="lower")
            for (y,x),v in np.ndenumerate(a):ax.text(x,y,f"{v:.0f}" if v in [0,100] else f"{v:.1f}",ha="center",va="center",fontsize=10,color="white" if v>66 else INK)
            ax.set(xticks=range(5),yticks=range(5),xlabel="直接邻居C数",title=f"{name} · 本人{s}")
            if j==0:ax.set_ylabel("对角邻居C数")
            ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which="minor",color="white",lw=1.5)
            ax.tick_params(which="both",length=0,pad=5)
            for spine in ax.spines.values():spine.set_visible(False)
    f.colorbar(im,cax=f.add_axes([.94,.19,.014,.58]),format=PercentFormatter(100))
    return f


def repeat_agreement(states):
    f=page("重复一致性：DeepSeek目前尚未测量，不能记为100%",
           "GPT与Qwen每局面各有两条回复；以下按局面统计这两条回复的组合", "07",
           "两次相同并不证明模型稳定；C/D各一次不代表真实选择概率恰好是0.5。DeepSeek只有一答，重复一致性为NA。\n若要比较回答随机性，应先统一每局面的重复次数，并显式记录可比的生成设置。")
    for i,name in enumerate(MODELS):
        x=.085+i*.31
        f.text(x,.70,name,fontsize=18,color=COLORS[name],weight="bold")
        if name=="DeepSeek":value="NA";desc="每局面仅1条回复"
        else:
            count=int(states[name].isin([0,1]).sum());value=f"{count/512:.2%}";desc=f"{count}/512局面，两次回复一致"
        f.text(x,.615,value,fontsize=31,weight="bold",color=COLORS[name]);f.text(x,.56,desc,fontsize=12,color=MUTED)
    ax=f.add_axes([.13,.23,.78,.23]);left=np.zeros(2)
    types=[(0,"两次均D","#94A6B5"),(.5,"C/D各一次","#E4AF64"),(1,"两次均C","#009E91")]
    for value,label,color in types:
        counts=np.array([sum(states[n]==value) for n in ["GPT","Qwen"]])
        ax.barh([1,0],counts,left=left,height=.48,color=color,label=label)
        for y,n,offset in zip([1,0],counts,left):ax.text(offset+n/2,y,str(n),ha="center",va="center",color=INK if value==.5 else "white",fontsize=13)
        left+=counts
    ax.set(yticks=[1,0],yticklabels=["GPT","Qwen"],xlim=(0,512),xlabel="局面数量",xticks=[0,128,256,384,512]);clean(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.28),ncol=3,frameon=False,fontsize=12)
    return f


def paired_choices(states):
    f=page("同局面对齐：保留另一张表的两次回答分歧",
           "行是DeepSeek的单次建议，列是GPT或Qwen的两答组合；格内数字为局面数，每个面板总计512", "08",
           "橙框表示对方两次回答一致，且均与DeepSeek方向相反：对GPT有85局面，对Qwen有144局面。\n中列保留两答不同的局面（GPT64、Qwen79），不强行判为C或D；计数不是模型总体胜负或稳定分歧率。")
    gs=f.add_gridspec(1,2,left=.10,right=.94,bottom=.27,top=.70,wspace=.28)
    arrays=[]
    for name in ["GPT","Qwen"]:
        arrays.append(np.array([[sum((states.DeepSeek==ds)&(states[name]==other)) for other in [0,.5,1]] for ds in [1,0]]))
    vmax=max(a.max() for a in arrays)
    for j,(name,a) in enumerate(zip(["GPT","Qwen"],arrays)):
        ax=f.add_subplot(gs[0,j]);ax.imshow(a,cmap=SEQ,vmin=0,vmax=vmax,aspect="auto")
        for (i,k),v in np.ndenumerate(a):ax.text(k,i,str(v),ha="center",va="center",fontsize=23,weight="bold",color="white" if v/vmax>.66 else INK)
        ax.set_xticks([0,1,2],["D / D","C / D","C / C"]);ax.set_yticks([0,1],["DeepSeek：C","DeepSeek：D"])
        ax.set_title(f"与{name}两条回复比较",fontsize=16,pad=15);ax.set_xlabel(f"{name}两答组合",labelpad=12)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True);ax.grid(which="minor",color="white",lw=2);ax.tick_params(which="both",length=0,pad=8)
        for y,x in [(0,0),(1,2)]:ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,edgecolor=COLORS["DeepSeek"],lw=3))
        for spine in ax.spines.values():spine.set_visible(False)
    return f


def make_figures(frames,states):
    return [("01_overview",overview(frames)),("02_transitions",transitions(frames)),("03_neighbors",neighbors(frames)),
            ("04_table_distance",table_distance(states)),("05_cases",cases(frames)),("06_diagonals",diagonals(states)),
            ("07_repeat_agreement",repeat_agreement(states)),("08_paired_choices",paired_choices(states))]


def export_tables(frames,states,report):
    folder=report/"tables";folder.mkdir(parents=True,exist_ok=True)
    overall=[];conditional=[];distance=[];pairs=[]
    for name,d in frames.items():
        d.to_csv(folder/(name+"_逐条回复.csv"),index=False,encoding="utf-8-sig")
        agreement=float(states[name].isin([0,1]).mean()) if REPEATS[name]>1 else np.nan
        overall.append(dict(model=name,states=512,replies=len(d),repeats=REPEATS[name],C=int(d.is_c.sum()),C_fraction=d.is_c.mean(),
                            C_given_current_C=d.loc[d.center=="C","is_c"].mean(),C_given_current_D=d.loc[d.center=="D","is_c"].mean(),
                            keep_current=d.recommendation.eq(d.center).mean(),two_reply_agreement=agreement))
        for (s,k),sub in d.groupby(["center","cardinal_c"]):
            conditional.append(dict(model=name,center=s,cardinal_c=k,states=sub.grid_key.nunique(),replies=len(sub),C=int(sub.is_c.sum()),C_fraction=sub.is_c.mean()))
    for a,b in PAIRS:
        for s in ["all","C","D"]:
            sub=states if s=="all" else states[states.center==s]
            distance.append(dict(model_a=a,model_b=b,center=s,states=len(sub),empirical_C_MAE_pp=(sub[a]-sub[b]).abs().mean()*100))
    for name in ["GPT","Qwen"]:
        for ds in [1,0]:
            for other in [0,.5,1]:pairs.append(dict(other=name,DeepSeek_action="C" if ds else "D",other_empirical_C=other,states=int(sum((states.DeepSeek==ds)&(states[name]==other)))))
    for rows,name in [(overall,"总体与本人条件统计"),(conditional,"四直接邻居条件统计"),(distance,"经验表距离"),(pairs,"DeepSeek对齐两答组合")]:
        pd.DataFrame(rows).to_csv(folder/(name+".csv"),index=False,encoding="utf-8-sig")
    states.to_csv(folder/"512局面对齐.csv",index=False,encoding="utf-8-sig")
    diagonal=states.groupby(["center","cardinal_c","diagonal_c"])[MODELS].mean().reset_index()
    diagonal["states"]=states.groupby(["center","cardinal_c","diagonal_c"]).size().to_numpy()
    diagonal.to_csv(folder/"直接与对角条件统计.csv",index=False,encoding="utf-8-sig")


def main():
    from three_advice_data import load_advice
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,default=REPORT);parser.add_argument("--font")
    args=parser.parse_args();report=args.report_dir.absolute()
    if report.is_relative_to(ROOT):report=WRITE_ROOT/report.relative_to(ROOT)
    font=setup_font(args.font)
    result=load_advice(report=report);frames,states=result["frames"],result["states"]
    export_tables(frames,states,report)
    figures=make_figures(frames,states);out=report/"figures";out.mkdir(parents=True,exist_ok=True)
    with PdfPages(report/"all_figures.pdf") as pdf:
        for name,f in figures:
            for ext in ["png","svg","pdf"]:f.savefig(out/(name+"."+ext),dpi=300)
            pdf.savefig(f);plt.close(f);print("Exported "+name,flush=True)
    (report/"绘图信息.json").write_text(json.dumps({"font":font,"matplotlib":matplotlib.__version__,"pandas":pd.__version__,
        "figures":len(figures),"states_per_model":512,"replies_per_state":REPEATS,
        "scope":"Empirical advice records only; no DeepSeek SPD simulation performed; no confidence intervals."},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":main()
