"""Visualize the 2026-10-27 Qwen scan, including its two observed replicates.

python tools/visualize_qwen_scan.py
Use --use-cache to reuse audited trajectories after verifying source hashes.
Read-only input: no simulation or LLM calls. See requirements-visualization.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_recent_results import setup_font, INK, MUTED, COLORS, GROUP_NAMES, clean
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(str(ROOT).lower())
if not (OUTPUT_ROOT.is_dir() and OUTPUT_ROOT.samefile(ROOT)):
    OUTPUT_ROOT = ROOT
BATCH = ROOT / "main/SPD/results/scan_20261027_122530_2次Qwen微扫描"
REPORT = OUTPUT_ROOT / "reports/qwen_20261027_repeats2"
BS = [1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2]
PS = [i / 10 for i in range(11)]
LAMS = [0., .1, .5, 1.]
SEEDS = [42, 43]
SEQ = LinearSegmentedColormap.from_list("qwen_coop", ["#F1F5F7", "#9AD7CC", "#15958B", "#17394F"])
DIV = LinearSegmentedColormap.from_list("qwen_gain", ["#B84E45", "#F9FBFC", "#148E88"])
SPREAD = LinearSegmentedColormap.from_list("qwen_spread", ["#F4F6F8", "#EFC98A", "#BA6B25"])


def page(title, subtitle, number, foot):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .958, "QWEN3.8  /  SPATIAL PRISONER'S DILEMMA  /  n = 2", fontsize=10, color=MUTED)
    fig.text(.055, .901, title, fontsize=26, weight="bold")
    fig.text(.055, .853, subtitle, fontsize=12.5, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .945], [.825, .825], transform=fig.transFigure, color="#DCE5EB", lw=1))
    fig.text(.055, .103, foot, fontsize=10.5, color=MUTED, va="top", linespacing=1.55)
    fig.text(.055, .025, "来源：2026-10-27 Qwen扫描  ·  seed 42 / 43  ·  同一离线建议表", fontsize=10, color=MUTED)
    fig.text(.945, .025, number, ha="right", fontsize=11, color=MUTED)
    return fig


def phase(ax, data, title, cmap, norm, fmt, fontsize=9.5):
    cmap = cmap.copy(); cmap.set_bad("#DDE2E7")
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(11), [f"{p:g}" for p in PS], fontsize=10)
    ax.set_yticks(range(7), [f"{b:g}" for b in BS], fontsize=11)
    ax.set_xticks(np.arange(-.5, 11, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="both", length=0, pad=6)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("建议资格比例 P", fontsize=12)
    for (i, j), value in np.ndenumerate(data):
        if np.isnan(value):
            label, color = "NA", MUTED
        else:
            rgba = im.cmap(im.norm(value))
            light = np.dot(rgba[:3], [.2126, .7152, .0722])
            label, color = fmt(value), "white" if light < .54 else INK
        ax.text(j, i, label, ha="center", va="center", fontsize=fontsize, color=color)
    for s in ax.spines.values(): s.set_visible(False)
    return im


def grid_values(df, metric, lam, reduction="mean"):
    sub = df[np.isclose(df.lambda_prob, lam)]
    grouped = sub.groupby(["b", "llm_access_fraction"])[metric]
    if reduction == "range":
        values = grouped.max() - grouped.min()
    else:
        values = grouped.agg(reduction)
    return values.unstack().reindex(index=BS, columns=PS).to_numpy(dtype=float)


def pct_label(v):
    if v == 0: return "0"
    if v < .1: return "<0.1"
    if abs(v - 100) < 1e-9: return "100"
    if v >= 99.95: return "≈100"
    return f"{v:.1f}"


def payoff_label(v):
    if abs(v) < .005: return "≈0" if v != 0 else "0"
    return f"{v:+.2f}"


def plot_maps(df):
    f = page("完整参数扫描：覆盖率与采纳率共同决定合作", "末1000步（9001–10000）合作率，两次运行均值；色标与格内数字单位均为%", "01",
             "P为固定建议资格比例，λ为资格者每次更新采用建议的概率；P=0或λ=0均为纯Fermi。\n每格仅2次，不能确定精确转变阈值；0与100为精确值，靠近两端的非精确值标为<0.1或≈100。")
    gs = f.add_gridspec(2, 2, left=.075, right=.90, bottom=.19, top=.77, wspace=.17, hspace=.48)
    for i, lam in enumerate(LAMS):
        ax = f.add_subplot(gs[i // 2, i % 2])
        im = phase(ax, grid_values(df, "tail_mean_cooperation_rate", lam) * 100,
                   "λ=0 · 纯Fermi" if lam == 0 else f"λ={lam:g}", SEQ, Normalize(0, 100), pct_label)
        if i % 2 == 0: ax.set_ylabel("背叛诱惑 b")
    cax = f.add_axes([.925, .19, .014, .58]); f.colorbar(im, cax=cax, format=PercentFormatter(100))
    return f


def plot_coverage(df):
    f = page("覆盖更多人通常有利，但λ更大并非处处更好", "线为两次均值；阴影为两次观测值之间的范围，圆点/三角分别为seed42/43", "02",
             "范围不是置信区间。P=0.9时，λ=0.5在所有b、两个seed均为100%合作，而λ=1均低于100%。\n此为已扫描点的观察，不据此将λ=0.5称为全局最优。")
    gs = f.add_gridspec(2, 4, left=.065, right=.96, bottom=.19, top=.77, wspace=.24, hspace=.43)
    for i, b in enumerate(BS):
        ax = f.add_subplot(gs[i // 4, i % 4])
        for lam in LAMS:
            sub = df[(df.b == b) & (df.lambda_prob == lam)]
            a = sub.pivot(index="llm_access_fraction", columns="seed", values="tail_mean_cooperation_rate").loc[PS, SEEDS].to_numpy()
            color = COLORS[lam]
            ax.fill_between(PS, a.min(axis=1), a.max(axis=1), color=color, alpha=.17, lw=0)
            ax.plot(PS, a.mean(axis=1), color=color, lw=1.9, ls="--" if lam == 0 else "-")
            for j, marker in enumerate(["o", "^"]):
                ax.plot(PS, a[:,j], linestyle="none", marker=marker, ms=3, color=color, alpha=.65)
        ax.set(xlim=(-.02, 1.02), ylim=(-.03, 1.05), title=f"b={b:g}", xticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0)); ax.tick_params(labelsize=10)
        if i % 4 == 0: ax.set_ylabel("末1000步合作率")
        if i >= 4: ax.set_xlabel("资格比例 P")
        clean(ax)
    ax = f.add_subplot(gs[1,3]); ax.axis("off")
    handles = [Line2D([], [], color=COLORS[l], lw=3, ls="--" if l == 0 else "-", label=f"λ={l:g}") for l in LAMS]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=14)
    ax.text(.03, .02, "圆点：seed42\n三角：seed43\n阴影：两次观测范围", fontsize=11, color=MUTED, transform=ax.transAxes, linespacing=1.5)
    return f


def plot_spread(df):
    f = page("两次运行的一致程度：差异集中在哪些区域？", "格内数字为 |seed42 − seed43| 的末段合作率差（百分点）；灰白接近0，橙色差异较大", "03",
             "最大差异出现在b=1.04、λ=0.5、P=0.3：两次为0%与11.410%。\n两次接近不等于已证实稳定，两次不同也不足以估计完整分布；后续重复应优先关注差异大的区域。")
    gs = f.add_gridspec(1,3,left=.065,right=.91,bottom=.25,top=.73,wspace=.22)
    vmax = max(grid_values(df, "tail_mean_cooperation_rate", l, "range").max() * 100 for l in LAMS[1:])
    vmax = max(1, np.ceil(vmax))
    for i, lam in enumerate(LAMS[1:]):
        ax = f.add_subplot(gs[0,i])
        im = phase(ax, grid_values(df,"tail_mean_cooperation_rate",lam,"range")*100, f"λ={lam:g}",
                   SPREAD,Normalize(0,vmax),lambda v: "0" if v==0 else ("<0.1" if v<.1 else f"{v:.1f}"),fontsize=8.5)
        if i==0: ax.set_ylabel("背叛诱惑 b")
    cax=f.add_axes([.935,.25,.014,.48]);f.colorbar(im,cax=cax,label="百分点")
    return f


def plot_welfare(df, group=False):
    if group:
        title="谁获得收益？以中等采纳率λ=0.5为例"
        sub="三个人群相对配对Fermi的全程每人每步收益差，两次均值；全体、资格组和无资格组共用色标"
        foot="分组按固定资格位置。P=0没有资格组，P=1没有无资格组：这些空组显示NA，不填0。\n每个seed先计算配对差，再求两次均值；均值的正负组合不代表每个seed都出现相同的组间分化。"
        columns=[("population",.5),("eligible",.5),("ineligible",.5)];idx="05"
    else:
        title="合作变化如何对应全体收益变化？"
        sub="全程每人每步收益配对差：建议实验 − 同初态纯Fermi；先逐seed配对，再平均两次结果"
        foot="收益取第1–10000步的轮末快照；图中用每人每步均值，便于阅读。正值表示相对对照获益。\n配对差包含邻域传播和演化路径改变；不是单次咨询的直接收益。近零细节见数据表。"
        columns=[("population",l) for l in LAMS[1:]];idx="04"
    f=page(title,sub,idx,foot)
    gs=f.add_gridspec(1,3,left=.065,right=.91,bottom=.25,top=.73,wspace=.22)
    arrays=[grid_values(df,f"delta_{g}_time_mean_payoff",l) for g,l in columns]
    lim=max(1,np.ceil(max(np.nanmax(np.abs(a)) for a in arrays)))
    for i,((g,l),a) in enumerate(zip(columns,arrays)):
        ax=f.add_subplot(gs[0,i]);im=phase(ax,a,GROUP_NAMES[g] if group else f"λ={l:g}",DIV,TwoSlopeNorm(0,-lim,lim),payoff_label,fontsize=8)
        if i==0:ax.set_ylabel("背叛诱惑 b")
    cax=f.add_axes([.935,.25,.014,.48]);f.colorbar(im,cax=cax)
    return f


def plot_equal_intensity(df):
    f=page("相同P×λ，也可能产生不同合作效果", "固定预期建议使用比例 P×λ=0.1；比较少数人高频采用与更多人低频采用", "06",
           "实线为两次均值，阴影为两次观测范围（非置信区间）；横轴等距列出扫描b值，不表示等数值间隔。\n在b=1.01、1.02、1.03三个已测值下，广覆盖低频方案合作率更高；b≥1.04三方案末段均为0。")
    gs=f.add_gridspec(1,2,left=.08,right=.95,bottom=.24,top=.69,wspace=.25)
    choices=[(.1,1.,"P=0.1，λ=1", "#287AB5"),(.2,.5,"P=0.2，λ=0.5","#009E91"),(1.,.1,"P=1，λ=0.1","#D67B2A")]
    for j,metric in enumerate(["tail_mean_cooperation_rate","delta_population_time_mean_payoff"]):
        ax=f.add_subplot(gs[0,j])
        for p,l,label,color in choices:
            arr=df[(df.llm_access_fraction==p)&(df.lambda_prob==l)].pivot(index="b",columns="seed",values=metric).loc[BS,SEEDS].to_numpy()
            x=np.arange(len(BS));ax.fill_between(x,arr.min(axis=1),arr.max(axis=1),color=color,alpha=.16,lw=0)
            ax.plot(x,arr.mean(axis=1),color=color,lw=2.4,marker="o",ms=6,label=label)
            ax.plot(x,arr[:,0],ls="none",marker=".",color=color,alpha=.55)
            ax.plot(x,arr[:,1],ls="none",marker="^",ms=4,color=color,alpha=.55)
        ax.set(xticks=range(7),xticklabels=[f"{b:g}" for b in BS],xlabel="背叛诱惑 b（离散扫描值）")
        if j==0:
            ax.set(ylim=(-.02, .77),ylabel="末1000步合作率")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.axhline(0,color=INK,lw=.8);ax.set_ylabel("Δ 全体每人每步收益")
        clean(ax)
    f.legend(*f.axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.52,.805),ncol=3,frameon=False,fontsize=14)
    return f


def plot_dynamics(trajectories):
    f=page("从早期演化到10000步：合作怎样建立或消失", "固定示例 b=1.02 / 1.1 / 1.2，P=0.2 / 0.5 / 0.9；线为两次均值，阴影为两次范围", "07",
           "横轴对数压缩（0–10步线性）；前200步逐步绘制，之后每10步取一个绘图点，统计仍使用全部步。\n仅展示选定参数切片；同一配对Fermi基线不增加重复数，末段不自动视为稳态。")
    gs=f.add_gridspec(3,3,left=.07,right=.95,bottom=.19,top=.73,wspace=.18,hspace=.42)
    take=np.unique(np.r_[np.arange(201),np.arange(210,10001,10)])
    for i,b in enumerate([1.02,1.1,1.2]):
        for j,p in enumerate([.2,.5,.9]):
            ax=f.add_subplot(gs[i,j])
            for l in LAMS:
                a=np.stack([trajectories[b,l,p,s] for s in SEEDS])[:,take]
                ax.fill_between(take,a.min(axis=0),a.max(axis=0),color=COLORS[l],alpha=.18,lw=0)
                ax.plot(take,a.mean(axis=0),color=COLORS[l],lw=1.5,ls="--" if l==0 else "-")
            ax.set_xscale("symlog",linthresh=10)
            ax.set(xlim=(0,10000),ylim=(-.02,1.04),xticks=[0,100,1000,10000],title=f"b={b:g}，P={p:g}")
            ax.xaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.tick_params(labelsize=9);ax.set_yticks([0,.5,1]);clean(ax)
            if j==0:ax.set_ylabel("合作率",fontsize=11)
            if i==2:ax.set_xlabel("时间步",fontsize=11)
    handles=[Line2D([],[],color=COLORS[l],lw=2.5,ls="--" if l==0 else "-",label=f"λ={l:g}") for l in LAMS]
    f.legend(handles=handles,loc="upper center",bbox_to_anchor=(.51,.825),ncol=4,frameon=False,fontsize=12)
    return f


def plot_advisor(advisor):
    f=page("Qwen建议更愿意在合作邻域中维持或引入合作", "512种完整3×3局面，每种2条回复；以下按四近邻合作数汇总展示，仿真仍匹配完整九格", "08",
           "比例来自等权枚举的局面，不是仿真实际访问频率。每局面只有2条回复；仿真两次重复是另一层随机性。\n全C与全D局面均获得保持原状态的建议；这些条件行为是机制线索，不能单独证明宏观结果的成因。")
    ax=f.add_axes([.09,.23,.55,.52])
    for state,color,marker in [("C","#009E91","o"),("D","#D67B2A","s")]:
        sub=advisor[advisor.center==state].groupby("cardinal_c").is_c.mean().reindex(range(5))
        ax.plot(range(5),sub,marker=marker,color=color,lw=2.7,ms=8,label=f"本人当前为{state}")
        for x,v in sub.items():
            ax.annotate(f"{v:.1%}",(x,v),xytext=(0,11 if state=="C" else -20),textcoords="offset points",ha="center",fontsize=12,color=color)
    ax.set(xlim=(-.15,4.15),ylim=(-.06,1.12),xticks=range(5),xlabel="四个直接邻居中的合作人数",ylabel="建议合作C的比例")
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,loc="upper left");clean(ax)
    agreement=advisor.groupby("grid_key").recommendation.nunique().eq(1).mean()
    f.text(.72,.69,"等权局面的合作建议率",fontsize=15)
    f.text(.72,.61,f"{advisor.is_c.mean():.2%}",fontsize=32,color="#008F87",weight="bold")
    f.text(.72,.48,"两条回答一致的局面",fontsize=15)
    f.text(.72,.40,f"{agreement:.2%}",fontsize=28,weight="bold")
    f.text(.72,.27,"所有b共用同一建议表；\n提示未给出具体数值b。",fontsize=13,color=MUTED,linespacing=1.7)
    return f


def plot_spatial(grids):
    f=page("最终空间状态：固定seed42的四种干预配置", "第10000步真实网格；列比较P=0.2 / 0.9、λ=0.5 / 1，行比较b=1.02 / 1.1 / 1.2", "09",
           "青绿为合作C，浅灰为背叛D；标签是该单次运行的终点合作率，不是末段均值或两次平均。\n固定seed42便于对照，不将此空间示例称为典型状态。")
    gs=f.add_gridspec(3,4,left=.15,right=.88,bottom=.17,top=.76,wspace=.16,hspace=.25)
    cmap=ListedColormap(["#E8EDF2","#008F87"])
    for i,b in enumerate([1.02,1.1,1.2]):
        for j,(p,l) in enumerate([(.2,.5),(.2,1.),(.9,.5),(.9,1.)]):
            ax=f.add_subplot(gs[i,j]);g=grids[b,l,p,42]
            ax.imshow(g,cmap=cmap,vmin=0,vmax=1,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values():s.set_visible(False)
            if i==0:ax.set_title(f"P={p:g}，λ={l:g}",fontsize=13,pad=12)
            if j==0:ax.set_ylabel(f"b={b:g}",fontsize=14,rotation=0,labelpad=42,va="center")
            ax.text(1.05,.5,f"{g.mean():.1%}",transform=ax.transAxes,va="center",fontsize=11,color=MUTED)
    return f


def export_tables(df, report):
    folder=report/"tables";folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/"逐次运行.csv",index=False,encoding="utf-8-sig")
    metrics=["tail_mean_cooperation_rate","final_cooperation_rate","delta_tail_mean_cooperation_rate"]
    metrics += [f"delta_{g}_{window}_mean_payoff" for g in GROUP_NAMES for window in ["early","time","tail"]]
    records=[]
    for keys,sub in df.groupby(["b","lambda_prob","llm_access_fraction"]):
        for metric in metrics:
            v=sub[metric]
            records.append(dict(zip(["b","lambda_prob","P"],keys),metric=metric,n=int(v.count()),
                                mean=v.mean(),min=v.min(),max=v.max(),range=v.max()-v.min(),
                                seed42=sub.loc[sub.seed==42,metric].iloc[0],seed43=sub.loc[sub.seed==43,metric].iloc[0]))
    pd.DataFrame(records).to_csv(folder/"参数汇总_均值_两次原值_范围.csv",index=False,encoding="utf-8-sig")
    for lam in LAMS:
        pd.DataFrame(grid_values(df,"tail_mean_cooperation_rate",lam)*100,index=BS,columns=PS).rename_axis("b").to_csv(
            folder/f"合作率百分比_lambda{lam:g}.csv",encoding="utf-8-sig")
    alive=df.copy();alive["alive"]=alive.final_cooperation_rate>0
    alive.groupby(["b","lambda_prob","llm_access_fraction"]).alive.agg(["sum","count"]).to_csv(
        folder/"终点仍有合作次数.csv",encoding="utf-8-sig")
    same=df[((df.llm_access_fraction==.1)&(df.lambda_prob==1))|((df.llm_access_fraction==.2)&(df.lambda_prob==.5))|((df.llm_access_fraction==1)&(df.lambda_prob==.1))]
    same.to_csv(folder/"相同P乘lambda对比_逐次.csv",index=False,encoding="utf-8-sig")
    mask=(df.llm_access_fraction>0)&(df.llm_access_fraction<1)&(df.lambda_prob>0)
    flags=df.loc[mask,["b","lambda_prob","llm_access_fraction","seed"]].copy()
    for window in ["early","time","tail"]:
        eligible=df.loc[mask,f"delta_eligible_{window}_mean_payoff"]
        pop=df.loc[mask,f"delta_population_{window}_mean_payoff"]
        flags[f"eligible_gain_population_loss_{window}"]=(eligible>0)&(pop<0)
        flags[f"delta_eligible_{window}"]=eligible;flags[f"delta_population_{window}"]=pop
    flags.to_csv(folder/"资格组增益_全体受损_逐次核对.csv",index=False,encoding="utf-8-sig")


def plot_horizon(trajectories):
    f=page("延长观察改变判断：5000步时的合作未必能够维持",
           "本次轨迹内的时间窗对比；选取4001–5000至9001–10000两次均值变化最大的两个参数组", "10",
           "左组两次分别在6315、9057步灭绝；右组seed43在8169步灭绝，seed42终点仍有4.52%合作。\n左组seed43末段均值为0.0316%，但终点已全D：末段平均非零不等于最终存续。每组仍只有2次。")
    gs=f.add_gridspec(2,2,left=.08,right=.95,bottom=.20,top=.75,wspace=.23,hspace=.55,
                      height_ratios=[1.3,1])
    take=np.unique(np.r_[np.arange(201),np.arange(210,10001,10)])
    for j,(b,l,p) in enumerate([(1.05,.5,.4),(1.03,1.,.1)]):
        ax=f.add_subplot(gs[0,j]);bar=f.add_subplot(gs[1,j])
        for si,(seed,color) in enumerate(zip(SEEDS,["#287AB5","#D67B2A"])):
            a=trajectories[b,l,p,seed]
            ax.plot(take,a[take],color=color,lw=1.15,alpha=.85,label=f"seed{seed}")
            windows=[a[4001:5001].mean(),a[9001:10001].mean()]
            positions=np.array([0.,1.])+(si-.5)*.28
            bar.bar(positions,windows,width=.26,color=color,alpha=.85)
            for x,y in zip(positions,windows):
                bar.text(x,y+.003,f"{y:.2%}",ha="center",va="bottom",fontsize=11,color=color)
        for start,end in [(4001,5000),(9001,10000)]:
            ax.axvspan(start,end,color="#9AB8C6",alpha=.18,lw=0)
        ax.set(xlim=(0,10000),ylim=(-.01,.55),xticks=[0,5000,10000],
               title=f"b={b:g}，P={p:g}，λ={l:g}",xlabel="时间步",ylabel="单次合作率")
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));clean(ax)
        if j==0:ax.legend(frameon=False,ncol=2,loc="upper right",fontsize=10)
        bar.set(xticks=[0,1],xticklabels=["4001–5000步","9001–10000步"],
                ylim=(0,.18),ylabel="1000步窗口均值")
        bar.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));clean(bar)
    return f


def main():
    from qwen_scan_data import load_scan
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir",type=Path,default=BATCH)
    parser.add_argument("--report-dir",type=Path,default=REPORT)
    parser.add_argument("--use-cache",action="store_true")
    parser.add_argument("--font")
    args=parser.parse_args();report=args.report_dir.absolute()
    if report.is_relative_to(ROOT):
        report=OUTPUT_ROOT/report.relative_to(ROOT)
    font=setup_font(args.font)
    config,df,trajectories,grids,advisor,audit=load_scan(args.batch_dir.resolve(),report,args.use_cache)
    export_tables(df,report)
    figures=[("01_cooperation",plot_maps(df)),("02_coverage",plot_coverage(df)),
             ("03_repeat_spread",plot_spread(df)),("04_welfare",plot_welfare(df)),
             ("05_group_welfare",plot_welfare(df,True)),("06_equal_intensity",plot_equal_intensity(df)),
             ("07_dynamics",plot_dynamics(trajectories)),("08_advisor",plot_advisor(advisor)),
             ("09_spatial",plot_spatial(grids)),("10_horizon",plot_horizon(trajectories))]
    folder=report/"figures";folder.mkdir(parents=True,exist_ok=True)
    with PdfPages(report/"all_figures.pdf") as pdf:
        for name,fig in figures:
            for suffix in ["png","svg","pdf"]:fig.savefig(folder/(name+"."+suffix),dpi=300)
            pdf.savefig(fig);plt.close(fig);print("Exported "+name,flush=True)
    (report/"绘图信息.json").write_text(json.dumps({"font":font,"matplotlib":matplotlib.__version__,
        "pandas":pd.__version__,"figures":len(figures),"model_label":"Qwen3.8","replicates":2,
        "range_semantics":"min-max of two observed seeds; not confidence intervals"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":main()
