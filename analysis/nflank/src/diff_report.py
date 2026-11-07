import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0,"manuscript/analysis"); sys.path.insert(0,"analysis/tss_strata/src")
import stats_gradient as SG
from manifest import TAG, checkpoint
OUT="manuscript/analysis/out"
BASE=["DeepSEA","DeepFormer","DeepATT"]; MODELS=BASE+["FunFormer_1k","FunFormer_20k"]

print("="*86); print("TABLE 1 / FIGURE 2a,b  — within-species mean AUROC (AUPRC)"); print("="*86)
rows=[]
for sp in sorted(TAG):
    for m in MODELS:
        ck,_=checkpoint(sp,m); exp=ck.split("/")[1]
        o=pd.read_csv(f"eval_results/test_set/{exp}_per_track.csv")
        n=pd.read_csv(f"eval_results/test_set_final/{exp}_per_track.csv")
        rows.append(dict(species=sp,model=m,tracks_old=o.auroc.notna().sum(),
            tracks_new=n.auroc.notna().sum(),
            auroc_old=o.auroc.mean(),auroc_new=n.auroc.mean(),
            auprc_old=o.auprc.mean(),auprc_new=n.auprc.mean()))
t=pd.DataFrame(rows); t["d_auroc"]=t.auroc_new-t.auroc_old; t["d_auprc"]=t.auprc_new-t.auprc_old
print(t.round(4).to_string(index=False))
print("\n-- architecture / context per species (pp) --")
w=t.pivot(index="species",columns="model",values=["auroc_old","auroc_new"])
for sp in w.index:
    o_bb=max(w.loc[sp,("auroc_old",b)] for b in BASE); n_bb=max(w.loc[sp,("auroc_new",b)] for b in BASE)
    oa=(w.loc[sp,("auroc_old","FunFormer_1k")]-o_bb)*100; na=(w.loc[sp,("auroc_new","FunFormer_1k")]-n_bb)*100
    oc=(w.loc[sp,("auroc_old","FunFormer_20k")]-w.loc[sp,("auroc_old","FunFormer_1k")])*100
    nc=(w.loc[sp,("auroc_new","FunFormer_20k")]-w.loc[sp,("auroc_new","FunFormer_1k")])*100
    print(f"  {sp:10s} architecture {oa:+6.2f} -> {na:+6.2f}    context {oc:+6.2f} -> {nc:+6.2f}")

print("\n"+"="*86); print("FIGURE 3 / TABLE 2 / S4 — cross-species"); print("="*86)
o=pd.read_csv(f"{OUT}/cross_species_margins_nofloor.csv"); n=pd.read_csv(f"{OUT}/cross_species_margins_final.csv")
po,pn=o[o.in_primary_set],n[n.in_primary_set]
for col in ("margin_arch_all","margin_all","margin_best1kb_all"):
    print(f"\n{col}")
    for b in ("mammal-mammal","mammal-chicken","tetrapod-fish"):
        print(f"  {b:16s} {po[po.bin==b][col].mean()*100:+7.2f} -> {pn[pn.bin==b][col].mean()*100:+7.2f} pp")
    rng=np.random.default_rng(0)
    so=SG.species_label_perm(po[["source","target",col]],col); sn=SG.species_label_perm(pn[["source","target",col]],col)
    _,pdo=SG.pearson_perm(po.divergence_mya,po[col],rng); _,pdn=SG.pearson_perm(pn.divergence_mya,pn[col],rng)
    print(f"  {'r':16s} {so['r']:+7.3f} -> {sn['r']:+7.3f}")
    print(f"  {'P species':16s} {so['p_r']:7.4f} -> {sn['p_r']:7.4f}")
    print(f"  {'P direction':16s} {pdo:7.5f} -> {pdn:7.5f}")
print("\n-- Table 2 per-assay (margin_arch, pp) --")
for b in ("mammal-mammal","mammal-chicken","tetrapod-fish"):
    cells=[]
    for a in ["ATAC","H3K27ac","H3K27me3","H3K4me1","H3K4me3"]:
        c=f"margin_arch_{a}"
        cells.append(f"{a} {po[po.bin==b][c].mean()*100:+.1f}->{pn[pn.bin==b][c].mean()*100:+.1f}")
    print(f"  {b:16s} " + "  ".join(cells))
