"""Plot the audited GPT scan and matched Qwen comparison; no simulation/API calls.

python tools/visualize_gpt_comparison.py --use-cache
Dependencies: tools/requirements-visualization.txt. Model names are local metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_recent_results import setup_font, INK, MUTED, COLORS, GROUP_NAMES, clean
from visualize_qwen_scan import BS, PS, LAMS, SEEDS, phase, grid_values, pct_label, payoff_label
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(str(ROOT).lower())
if not (OUTPUT_ROOT.is_dir() and OUTPUT_ROOT.samefile(ROOT)):
    OUTPUT_ROOT = ROOT
REPORT = OUTPUT_ROOT / "reports/gpt_20261027_repeats2"
KEYS = ["b", "lambda_prob", "llm_access_fraction", "seed"]
GPT = "#287AB5"
QWEN = "#009E91"
SEQ = LinearSegmentedColormap.from_list("gpt_c", ["#F2F6FA", "#97CDE0", GPT, "#15384E"])
GAIN = LinearSegmentedColormap.from_list("gpt_gain", ["#C16B39", "#F9FBFC", GPT])
GAP = LinearSegmentedColormap.from_list("model_gap", [QWEN, "#F9FBFC", GPT])
SPREAD = LinearSegmentedColormap.from_list("gpt_spread", ["#F3F6F8", "#F1C785", "#AF6029"])


def page(title, subtitle, number, foot, comparison=False):
    f = plt.figure(figsize=(16, 9))
    tag = "GPT / QWEN · MATCHED SETTINGS" if comparison else "GPT SCAN · RECORDED MODEL: gpt-5.5"
    f.text(.055,.958,tag+"  /  FIXED ADVICE TABLES  /  n = 2",fontsize=10,color=MUTED)
    f.text(.055,.901,title,fontsize=25,weight="bold")
    f.text(.055,.853,subtitle,fontsize=12,color=MUTED)
    f.add_artist(plt.Line2D([.055,.945],[.825,.825],transform=f.transFigure,color="#DCE5EB",lw=1))
    f.text(.055,.103,foot,fontsize=10.5,color=MUTED,va="top",linespacing=1.55)
    f.text(.055,.025,"来源：2026-10-27参数扫描 · seed42 / 43 · 每模型一张固定建议表",fontsize=10,color=MUTED)
    f.text(.945,.025,number,ha="right",fontsize=11,color=MUTED)
    return f


def plot_gpt_maps(df):
    f=page("GPT完整参数扫描：高覆盖不一定带来高合作",
           "末1000步（9001–10000）合作率，两次均值；格内单位为%", "01",
           "P为固定建议资格比例，λ为资格者每次更新采用建议的概率；P=0或λ=0均为纯Fermi。\n0与100为精确值；非精确近端标<0.1或≈100。末段均值非零不等于终点仍有合作。")
    gs=f.add_gridspec(2,2,left=.075,right=.90,bottom=.19,top=.77,wspace=.17,hspace=.48)
    for i,l in enumerate(LAMS):
        ax=f.add_subplot(gs[i//2,i%2])
        im=phase(ax,grid_values(df,"tail_mean_cooperation_rate",l)*100,
                 "λ=0 · 纯Fermi" if l==0 else f"λ={l:g}",SEQ,Normalize(0,100),pct_label)
        if i%2==0:ax.set_ylabel("背叛诱惑 b")
    f.colorbar(im,cax=f.add_axes([.925,.19,.014,.58]),format=PercentFormatter(100))
    return f


def plot_gpt_coverage(df):
    f=page("完全采纳建议时，覆盖率上升可能伴随合作骤降",
           "线为两次均值，阴影为两次观测范围；圆点/三角分别为seed42/43", "02",
           "例如b=1.02、λ=1：P=0.6时末段均值89.72%，P=0.7时仅1.67%；两次都出现这种排序。\n连线仅辅助阅读离散扫描点；范围不是置信区间，不能据此确定精确转变阈值。")
    gs=f.add_gridspec(2,4,left=.065,right=.96,bottom=.19,top=.77,wspace=.24,hspace=.43)
    for i,b in enumerate(BS):
        ax=f.add_subplot(gs[i//4,i%4])
        for l in LAMS:
            a=df[(df.b==b)&(df.lambda_prob==l)].pivot(index="llm_access_fraction",columns="seed",values="tail_mean_cooperation_rate").loc[PS,SEEDS].to_numpy()
            ax.fill_between(PS,a.min(1),a.max(1),color=COLORS[l],alpha=.17,lw=0)
            ax.plot(PS,a.mean(1),color=COLORS[l],lw=1.9,ls="--" if l==0 else "-")
            for j,m in enumerate(["o","^"]):ax.plot(PS,a[:,j],ls="none",marker=m,ms=3,color=COLORS[l],alpha=.65)
        ax.set(xlim=(-.02,1.02),ylim=(-.03,1.05),title=f"b={b:g}",xticks=[0,.5,1])
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.tick_params(labelsize=10);clean(ax)
        if i%4==0:ax.set_ylabel("末1000步合作率")
        if i>=4:ax.set_xlabel("资格比例 P")
    ax=f.add_subplot(gs[1,3]);ax.axis("off")
    ax.legend(handles=[Line2D([],[],color=COLORS[l],lw=3,ls="--" if l==0 else "-",label=f"λ={l:g}") for l in LAMS],loc="upper left",frameon=False,fontsize=14)
    ax.text(.03,.02,"圆点：seed42\n三角：seed43\n阴影：两次观测范围",transform=ax.transAxes,fontsize=11,color=MUTED,linespacing=1.5)
    return f


def triple(df,metric,title,subtitle,number,foot,cmap,fmt,scale=1,reduction="mean",comparison=False,group=False):
    f=page(title,subtitle,number,foot,comparison)
    gs=f.add_gridspec(1,3,left=.065,right=.91,bottom=.25,top=.73,wspace=.22)
    columns=[(f"delta_{g}_time_mean_payoff",1.,GROUP_NAMES[g]) for g in GROUP_NAMES] if group else [(metric,l,f"λ={l:g}") for l in LAMS[1:]]
    arrays=[grid_values(df,m,l,reduction)*scale for m,l,_ in columns]
    lim=max(1,np.ceil(max(np.nanmax(np.abs(a)) for a in arrays)))
    norm=Normalize(0,lim) if reduction=="range" else TwoSlopeNorm(0,-lim,lim)
    for i,(a,(_,_,label)) in enumerate(zip(arrays,columns)):
        ax=f.add_subplot(gs[0,i]);im=phase(ax,a,label,cmap,norm,fmt,fontsize=8)
        if i==0:ax.set_ylabel("背叛诱惑 b")
    f.colorbar(im,cax=f.add_axes([.935,.25,.014,.48]))
    return f


def gap_label(v):
    if v==0:return "0"
    if abs(v)<.1:return "≈0"
    return f"{v:+.1f}"


def plot_dynamics(gt,qt):
    f=page("合作差异如何形成：同参数下的演化轨迹",
           "GPT与Qwen均值及两次范围，并列共同Fermi基线；选取覆盖骤降、重复分歧和全采纳三个诊断场景", "06",
           "横轴对数压缩（0–10步线性）；前200步逐步绘制，其后每10步取点，统计使用全部步。\n这些是事后诊断切片；阴影不是置信区间，均值曲线不是单次运行，末段不自动代表稳态。",True)
    cases=[(1.01,1.,.6),(1.01,1.,.7),(1.02,1.,.6),(1.02,1.,.7),(1.03,.5,1.),(1.2,1.,1.)]
    gs=f.add_gridspec(2,3,left=.07,right=.95,bottom=.20,top=.73,wspace=.19,hspace=.43)
    take=np.unique(np.r_[np.arange(201),np.arange(210,10001,10)])
    for i,(b,l,p) in enumerate(cases):
        ax=f.add_subplot(gs[i//3,i%3])
        for data,lam,color,label in [(gt,0.,"#7A8794","Fermi"),(gt,l,GPT,"GPT"),(qt,l,QWEN,"Qwen")]:
            a=np.stack([data[b,lam,p,s] for s in SEEDS])[:,take]
            ax.fill_between(take,a.min(0),a.max(0),color=color,alpha=.16,lw=0)
            ax.plot(take,a.mean(0),color=color,lw=1.6,ls="--" if lam==0 else "-",label=label)
        ax.set_xscale("symlog",linthresh=10)
        ax.set(xlim=(0,10000),ylim=(-.02,1.04),xticks=[0,100,1000,10000],yticks=[0,.5,1],title=f"b={b:g}，P={p:g}，λ={l:g}")
        ax.xaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.tick_params(labelsize=9);clean(ax)
        if i%3==0:ax.set_ylabel("合作率")
        if i>=3:ax.set_xlabel("时间步")
    f.legend(*f.axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.51,.825),ncol=3,frameon=False,fontsize=13)
    return f


def plot_advisor(ga,qa):
    f=page("建议表差异：合作邻域中的背叛者是否愿意转向合作？",
           "同一提示、同样512种完整3×3局面、每局面2条回复；下图按本人状态与四近邻合作数汇总", "07",
           f"等权枚举的合作建议率：GPT {ga.is_c.mean():.2%}，Qwen {qa.is_c.mean():.2%}；这不是仿真实际访问频率。\n两表全C均建议C、全D均建议D。条件响应差异是机制线索；不能单凭此图证明宏观差异的原因。",True)
    gs=f.add_gridspec(1,2,left=.08,right=.95,bottom=.23,top=.73,wspace=.24)
    for i,state in enumerate(["C","D"]):
        ax=f.add_subplot(gs[0,i])
        for data,color,name,dy in [(ga,GPT,"GPT",-22),(qa,QWEN,"Qwen",12)]:
            s=data[data.center==state].groupby("cardinal_c").is_c.mean().reindex(range(5))
            ax.plot(range(5),s,color=color,lw=2.6,marker="o",ms=7,label=name)
            for x,y in s.items():ax.annotate(f"{y:.1%}",(x,y),xytext=(0,dy),textcoords="offset points",ha="center",color=color,fontsize=11)
        ax.set(xlim=(-.15,4.15),ylim=(-.11,1.13),xticks=range(5),xlabel="四个直接邻居中的合作人数",ylabel="建议合作C的比例",title=f"本人当前为{state}")
        ax.yaxis.set_major_formatter(PercentFormatter(1));clean(ax)
    f.legend(*f.axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.51,.82),ncol=2,frameon=False,fontsize=14)
    return f


def plot_equal(g,q):
    f=page("同样P×λ=0.1，GPT与Qwen的覆盖方式效应有何不同？",
           "每模型分别比较三种固定预期建议使用比例的配置；线为两次均值，阴影为两次范围", "10",
           "横轴等距列出离散b值，不表示相等数值间隔。两模型b≥1.04时，这三种方案末段合作均为0。\nGPT的两个seed排序并不一致；不能只凭均值的小幅排序就认定广覆盖更好。",True)
    choices=[(.1,1.,"P=0.1，λ=1","#287AB5"),(.2,.5,"P=0.2，λ=0.5","#009E91"),(1.,.1,"P=1，λ=0.1","#D67B2A")]
    gs=f.add_gridspec(1,2,left=.08,right=.95,bottom=.24,top=.69,wspace=.23)
    for i,(df,name) in enumerate([(g,"GPT"),(q,"Qwen")]):
        ax=f.add_subplot(gs[0,i])
        for p,l,label,color in choices:
            a=df[(df.llm_access_fraction==p)&(df.lambda_prob==l)].pivot(index="b",columns="seed",values="tail_mean_cooperation_rate").loc[BS,SEEDS].to_numpy()
            x=np.arange(7);ax.fill_between(x,a.min(1),a.max(1),color=color,alpha=.16,lw=0)
            ax.plot(x,a.mean(1),color=color,lw=2.3,marker="o",ms=5,label=label)
            for s,m in enumerate([".","^"]):ax.plot(x,a[:,s],ls="none",marker=m,ms=5,color=color,alpha=.65)
        ax.set(xticks=range(7),xticklabels=[f"{b:g}" for b in BS],ylim=(-.02,.77),xlabel="背叛诱惑 b（离散扫描值）",ylabel="末1000步合作率",title=name)
        ax.yaxis.set_major_formatter(PercentFormatter(1));clean(ax)
    f.legend(*f.axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.52,.805),ncol=3,frameon=False,fontsize=13)
    return f


def plot_survival(g,q):
    f=page("终点是否还有合作：逐格列出两次运行的存续次数",
           "统计第10000步合作人数>0的运行次数；0/2、1/2、2/2分别表示零次、一次、两次仍有合作", "11",
           "活跃干预范围（P>0、λ>0）内：GPT终点全D为266/420次，Qwen为158/420次；两批均无全D后恢复事件。\n这些计数描述当前扫描网格；每格只有2次，不能把0/2或2/2当成真实存续概率为0或1。",True)
    cmap=ListedColormap(["#EDF1F4","#E7BC75","#008F87"]);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    gs=f.add_gridspec(2,3,left=.065,right=.96,bottom=.19,top=.77,wspace=.21,hspace=.5)
    for i,(df,name) in enumerate([(g,"GPT"),(q,"Qwen")]):
        a=df.copy();a["alive"]=a.final_cooperation_rate>0
        for j,l in enumerate(LAMS[1:]):
            ax=f.add_subplot(gs[i,j]);v=grid_values(a,"alive",l,"sum")
            phase(ax,v,f"{name} · λ={l:g}",cmap,norm,lambda n:f"{int(n)}/2",fontsize=8.5)
            if j==0:ax.set_ylabel("背叛诱惑 b")
    return f


def plot_spatial(gg,qg):
    f=page("最终空间状态：同参数、同种子，对照两张建议表",
           "第10000步真实50×50网格；每个参数组同时展示GPT与Qwen的两个seed", "12",
           "青绿为合作C，浅灰为背叛D。每张图的百分比是该单次终点合作率，不是末段均值。\n示例用于检查空间形态，不把两个种子或选定切片视为模型表现的完整分布。",True)
    cases=[(1.02,1.,.6),(1.02,1.,.7),(1.2,1.,1.)]
    gs=f.add_gridspec(3,4,left=.20,right=.91,bottom=.17,top=.76,wspace=.19,hspace=.30)
    cmap=ListedColormap(["#E8EDF2","#008F87"])
    for i,(b,l,p) in enumerate(cases):
        for j,(data,name,s) in enumerate([(gg,"GPT",42),(gg,"GPT",43),(qg,"Qwen",42),(qg,"Qwen",43)]):
            ax=f.add_subplot(gs[i,j]);v=data[b,l,p,s]
            ax.imshow(v,cmap=cmap,vmin=0,vmax=1,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if i==0:ax.set_title(f"{name} · seed{s}",fontsize=12,pad=12)
            if j==0:ax.set_ylabel(f"b={b:g}\nP={p:g}，λ={l:g}",fontsize=12,rotation=0,labelpad=65,va="center")
            ax.text(1.04,.5,f"{v.mean():.1%}",transform=ax.transAxes,va="center",fontsize=10.5,color=MUTED)
    return f


def paired_frame(g,q):
    assert len(g)==len(q)==616
    assert set(map(tuple,g[KEYS].to_numpy()))==set(map(tuple,q[KEYS].to_numpy()))
    m=g.merge(q,on=KEYS,how="inner",suffixes=("_gpt","_qwen"),validate="one_to_one").copy()
    assert len(m)==616
    metrics=["tail_mean_cooperation_rate","final_cooperation_rate","time_mean_cooperation_rate"]
    metrics += [f"{prefix}{group}_{window}_mean_payoff" for prefix in ["","delta_"] for group in GROUP_NAMES for window in ["early","time","tail"]]
    differences={"gap_"+metric:m[metric+"_gpt"]-m[metric+"_qwen"] for metric in metrics}
    m=pd.concat([m,pd.DataFrame(differences,index=m.index)],axis=1)
    assert np.allclose(m.gap_population_time_mean_payoff,m.gap_delta_population_time_mean_payoff,atol=1e-12)
    return m


def export_tables(g,q,m,report):
    folder=report/"tables";folder.mkdir(parents=True,exist_ok=True)
    metrics=["tail_mean_cooperation_rate","final_cooperation_rate","delta_tail_mean_cooperation_rate"]
    metrics += [f"delta_{group}_{window}_mean_payoff" for group in GROUP_NAMES for window in ["early","time","tail"]]
    for df,name in [(g,"GPT"),(q,"Qwen"),(m,"GPT减Qwen")]:
        df.to_csv(folder/(name+"_逐次运行.csv"),index=False,encoding="utf-8-sig")
        cols=[c for c in m if c.startswith("gap_")] if name=="GPT减Qwen" else metrics
        rows=[]
        for keys,sub in df.groupby(KEYS[:-1]):
            for metric in cols:
                v=sub[metric]
                rows.append(dict(zip(KEYS[:-1],keys),metric=metric,n=int(v.count()),mean=v.mean(),min=v.min(),max=v.max(),
                                 range=v.max()-v.min(),seed42=sub.loc[sub.seed==42,metric].iloc[0],seed43=sub.loc[sub.seed==43,metric].iloc[0]))
        pd.DataFrame(rows).to_csv(folder/(name+"_均值_原值_范围.csv"),index=False,encoding="utf-8-sig")
    states=[]
    for key,sub in m[(m.llm_access_fraction>0)&(m.lambda_prob>0)].groupby(KEYS[:-1]):
        v=sub.gap_tail_mean_cooperation_rate.to_numpy()
        label="GPT两次更高" if (v>0).all() else ("Qwen两次更高" if (v<0).all() else ("两次相等" if (v==0).all() else ("方向相反" if v.min()<0<v.max() else "一次相等")))
        states.append(dict(zip(KEYS[:-1],key),comparison=label,seed42_pp=sub.loc[sub.seed==42,"gap_tail_mean_cooperation_rate"].iloc[0]*100,
                           seed43_pp=sub.loc[sub.seed==43,"gap_tail_mean_cooperation_rate"].iloc[0]*100))
    pd.DataFrame(states).to_csv(folder/"合作差异方向_210个活跃参数组.csv",index=False,encoding="utf-8-sig")
    survival=pd.concat([df[KEYS+["final_cooperation_rate"]].assign(model=name,alive=df.final_cooperation_rate>0)
                        for df,name in [(g,"GPT"),(q,"Qwen")]])
    survival.groupby(["model"]+KEYS[:-1]).alive.agg(["sum","count"]).to_csv(folder/"终点合作存续次数.csv",encoding="utf-8-sig")


def make_figures(g,q,gt,qt,gg,qg,ga,qa):
    m=paired_frame(g,q)
    return m,[
        ("01_gpt_cooperation",plot_gpt_maps(g)),
        ("02_gpt_coverage",plot_gpt_coverage(g)),
        ("03_gpt_welfare",triple(g,"delta_population_time_mean_payoff","GPT是否改善全体收益？与配对Fermi逐次比较",
            "第1–10000步每人每步收益差的两次均值；蓝色为获益、橙色为受损", "03",
            "先逐seed计算建议实验−对应Fermi，再取两次均值；收益排除第0步。近零细节见数据表。\n合作与收益指标不能互相替代；差值包含邻居传播及整段路径改变，不是单次咨询的直接收益。",GAIN,payoff_label)),
        ("04_cooperation_gap",triple(m,"gap_tail_mean_cooperation_rate","同参数对比：Qwen在多数有效干预组合中合作更高",
            "GPT − Qwen 的末段合作率差（百分点），先按同参数同seed相减，再取两次均值", "04",
            "负值（青绿）表示Qwen更高，正值（蓝）表示GPT更高。210个活跃参数组中，126组两次都是Qwen更高。\n没有一组两次都是GPT更高；另有80组两次相等、3组一次相等、1组方向相反。仅比较本次两张固定表。",GAP,gap_label,100,comparison=True)),
        ("05_payoff_gap",triple(m,"gap_population_time_mean_payoff","收益对比：多数配置的Qwen全程收益更高",
            "GPT − Qwen 的全程每人每步收益差，两次均值；负值青绿、正值蓝", "05",
            "相同参数、相同seed的Fermi对照一致，因此模型收益差也等于两模型相对Fermi增量之差。\n210组中190组两次Qwen更高、3组两次GPT略高、17组方向相反；差值近零不代表完全相同。",GAP,payoff_label,comparison=True)),
        ("06_dynamics",plot_dynamics(gt,qt)),
        ("07_advisor",plot_advisor(ga,qa)),
        ("08_gpt_repeat_spread",triple(g,"tail_mean_cooperation_rate","GPT两次重复：少数区域出现很大的存续分歧",
            "末段合作率的两次观测跨度 |seed42 − seed43|，单位为百分点", "08",
            "最大跨度：b=1.03、λ=0.5、P=1，一次为0%，另一次35.1429%；两次平均17.57%掩盖了分歧。\n两次接近不代表总体方差小；每格n=2，不能据此计算可靠置信区间或宣称显著差异。",SPREAD,lambda x:"0" if x==0 else ("<0.1" if x<.1 else f"{x:.1f}"),100,"range")),
        ("09_group_welfare",triple(g,"","GPT完全采纳建议时：三个群体如何受影响？",
            "固定λ=1；全体、资格组、无资格组相对配对Fermi的全程每人每步收益差，两次均值", "09",
            "按固定资格位置分组。P=0没有资格组，P=1没有无资格组；这些空组显示NA，不填0。\n先逐seed配对后求均值；均值之间的正负组合不表示每个seed都出现相同的群体利益分化。",GAIN,payoff_label,group=True)),
        ("10_equal_intensity",plot_equal(g,q)),
        ("11_survival",plot_survival(g,q)),
        ("12_spatial",plot_spatial(gg,qg)),
    ]


def main():
    from gpt_comparison_data import load_comparison
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,default=REPORT)
    parser.add_argument("--use-cache",action="store_true")
    parser.add_argument("--font")
    args=parser.parse_args();report=args.report_dir.absolute()
    if report.is_relative_to(ROOT):report=OUTPUT_ROOT/report.relative_to(ROOT)
    font=setup_font(args.font)
    bundle=load_comparison(use_cache=args.use_cache,report=report)
    gc,g,gt,gg,ga,gau=bundle["gpt"];qc,q,qt,qg,qa,qau=bundle["qwen"]
    m,figures=make_figures(g,q,gt,qt,gg,qg,ga,qa)
    export_tables(g,q,m,report)
    folder=report/"figures";folder.mkdir(parents=True,exist_ok=True)
    with PdfPages(report/"all_figures.pdf") as pdf:
        for name,f in figures:
            for suffix in ["png","svg","pdf"]:f.savefig(folder/(name+"."+suffix),dpi=300)
            pdf.savefig(f);plt.close(f);print("Exported "+name,flush=True)
    (report/"绘图信息.json").write_text(json.dumps({"font":font,"matplotlib":matplotlib.__version__,"pandas":pd.__version__,
        "figures":len(figures),"model_labels_from_local_metadata":["gpt-5.5","Qwen3.8-27B-f16.gguf"],"replicates":2,
        "gap_direction":"GPT minus Qwen at matching b, lambda, P, seed", "range_semantics":"min-max of two observed seeds; not confidence intervals"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(report,flush=True)


if __name__=="__main__":main()
