
import sys, os, re, math
sys.path.insert(0, "/home/herikristianto/src")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import config as cfg
from utils.data_utils import load_split_arrays, measure_latency_cpu, count_params
from utils.metrics import auroc, dual_threshold_eval, bootstrap_auroc_ci, wilcoxon_test, delong_test, holm_bonferroni
from models.teacher import build_teacher
from models.students import build_student
from utils.gwo import decode_position, LOWER, UPPER, PARAM_NAMES

# Ambil run_woa dari 07_baselines.py
import importlib.util, types
spec = importlib.util.spec_from_file_location("_b07", "/home/herikristianto/src/07_baselines.py")
_b07 = importlib.util.module_from_spec(spec)
# Patch _GPU sebelum load
import sys as _sys; _sys.modules["_b07"] = _b07
spec.loader.exec_module(_b07)
run_woa = _b07.run_woa
FeatureRegressor = _b07.FeatureRegressor
train_kd_baseline = _b07.train_kd_baseline


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")

# ── Hasil sudah selesai (dari log) ───────────────────────────────────────
PSO_SCORES = [0.7662, 0.7682, 0.7663, 0.7657, 0.7666]
GA_SCORES  = [0.7665, 0.7663, 0.7655, 0.7676, 0.7671]
WOA_SEEDS_DONE = {0: 0.7669, 1: 0.7668, 2: 0.7660}

# GWO per-seed scores
gwo_hist = pd.read_csv(cfg.RESULTS_DIR / "gwo_history.csv")
GWO_SCORES = [float(gwo_hist[gwo_hist["seed"]==s]["alpha_score"].max())
              for s in cfg.META_SEEDS]
log(f"GWO scores: {[f'{v:.4f}' for v in GWO_SCORES]}")

# ── Load teacher + data ke GPU ────────────────────────────────────────────
ckpt = torch.load(cfg.CHECKPOINTS_DIR/"teacher_best.pt", map_location=device, weights_only=False)
n_features = ckpt["n_features"]
teacher = build_teacher(ckpt["model_name"], n_features=n_features).to(device)
teacher.load_state_dict(ckpt["state_dict"]); teacher.eval()

X_train, y_train = load_split_arrays("train")
X_val,   y_val   = load_split_arrays("val")
X_test,  y_test  = load_split_arrays("test")
n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
pos_weight = torch.tensor(min(n_neg/n_pos,10.0)).to(device)

log("Loading data to GPU...")
_X  = torch.from_numpy(X_train).float().to(device)
_y  = torch.from_numpy(y_train.astype("float32")).to(device)
_Xv = torch.from_numpy(X_val).float().to(device)
_yv = torch.from_numpy(y_val.astype("float32")).to(device)
_Xt = torch.from_numpy(X_test).float().to(device)
_yt = torch.from_numpy(y_test.astype("float32")).to(device)
_pi = torch.where(_y==1)[0]; _ni = torch.where(_y==0)[0]
log(f"GPU ready: {torch.cuda.memory_allocated()/1e9:.1f} GB")

def g_train(bs=4096):
    half=bs//2; nb=len(_ni)//half
    pi=_pi[torch.randint(0,len(_pi),(nb*half,),device=device)]
    ni=_ni[torch.randperm(len(_ni),device=device)[:nb*half]]
    for i in range(nb):
        s=i*half; idx=torch.cat([pi[s:s+half],ni[s:s+half]]); yield _X[idx],_y[idx]
def g_val(bs=4096):
    for s in range(0,len(_yv),bs): yield _Xv[s:s+bs],_yv[s:s+bs]
def g_test(bs=4096):
    for s in range(0,len(_yt),bs): yield _Xt[s:s+bs],_yt[s:s+bs]

# ── Fitness function ──────────────────────────────────────────────────────
def fitness(params):
    s_idx = params["s"]
    s = build_student(s_idx, n_features=n_features).to(device)
    reg = FeatureRegressor(s.feature_dim, teacher.feature_dim).to(device)
    opt = torch.optim.Adam(list(s.parameters())+list(reg.parameters()), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    T, alpha, beta = params["T"], params["alpha_kd"], params["beta_kd"]
    s.train(); reg.train(); teacher.eval()
    for _ in range(5):
        for X_b,y_b in g_train():
            with torch.no_grad(): z_t,f_t = teacher(X_b)
            z_s,f_s = s(X_b); fp=reg(f_s)
            lt=torch.stack([-z_t,z_t],1)/T; ls=torch.stack([-z_s,z_s],1)/T
            L = alpha*bce(z_s,y_b) + (1-alpha)*(T**2)*F.kl_div(F.log_softmax(ls,1),F.softmax(lt,1).detach(),reduction="batchmean") + beta*F.mse_loss(fp,f_t.detach())
            opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(list(s.parameters())+list(reg.parameters()),1.0); opt.step()
    s.eval(); sc,lb=[],[]
    with torch.no_grad():
        for X_b,y_b in g_val():
            z_s,_=s(X_b); sc.append(torch.sigmoid(z_s).cpu().numpy()); lb.append(y_b.cpu().numpy())
    return auroc(np.concatenate(lb),np.concatenate(sc))

# ── WOA: lanjutkan seed 3 dan 4 ──────────────────────────────────────────
log("\n[WOA] Melanjutkan seed 3 dan 4...")
WOA_SCORES = list(WOA_SEEDS_DONE.values())  # [0.7669, 0.7668, 0.7660]
for seed in [3, 4]:
    log(f"  Seed {seed} ...")
    _, score, _ = run_woa(fitness, n_wolves=cfg.GWO_N_WOLVES,
                          i_max=cfg.GWO_I_MAX, seed=seed, verbose=True)
    WOA_SCORES.append(score)
    log(f"  → seed={seed} best score={score:.4f}")

log(f"  WOA scores: {[f'{s:.4f}' for s in WOA_SCORES]}")
log(f"  WOA mean={np.mean(WOA_SCORES):.4f} ± {np.std(WOA_SCORES):.4f}")

# ── Wilcoxon tests ────────────────────────────────────────────────────────
log("\n[Wilcoxon tests: GWO vs competitors]")
meta_scores = {"PSO": PSO_SCORES, "GA": GA_SCORES, "WOA": WOA_SCORES}
wilcox_rows=[]; p_vals=[]
for name, scores in meta_scores.items():
    wt = wilcoxon_test(GWO_SCORES, scores)
    p_vals.append(wt["p_value"])
    wilcox_rows.append({"comparison":f"GWO vs {name}","gwo_mean":np.mean(GWO_SCORES),
        "other_mean":np.mean(scores),"p_value":wt["p_value"],"significant":wt["significant"]})
    log(f"  GWO vs {name}: p={wt['p_value']:.4f} {'✓' if wt['significant'] else '✗'}")
from utils.metrics import holm_bonferroni
corr=holm_bonferroni(p_vals)
for i,r in enumerate(wilcox_rows): r["p_corrected"]=corr[i]
pd.DataFrame(wilcox_rows).to_csv(cfg.RESULTS_DIR/"wilcoxon_meta.csv",index=False)
log(f"  ✅ wilcoxon_meta.csv saved")

# ── KD Baselines ──────────────────────────────────────────────────────────
log("\n[KD Baselines: Vanilla, FitNet, AT]")
gwo_df = pd.read_csv(cfg.RESULTS_DIR/"gwo_kd_result.csv")
gwo_s = int(re.search(r"S(\d)", gwo_df["model"].iloc[0]).group(1))
log(f"  Student S{gwo_s}")
all_kd=[]; kd_rows=[]

for method in ["vanilla","fitnet","at"]:
    log(f"\n[{method.upper()}]")
    torch.manual_seed(cfg.RANDOM_SEED)
    student = build_student(gwo_s, n_features=n_features).to(device)
    reg2 = FeatureRegressor(student.feature_dim, teacher.feature_dim).to(device)
    bce2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt2 = torch.optim.Adam(list(student.parameters())+list(reg2.parameters()), lr=1e-3)
    sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2,"max",factor=0.5,patience=5,min_lr=1e-6)
    best_auc2=-1.; best_state2=None; pat2=0
    teacher.eval()
    for epoch in range(1,51):
        student.train(); reg2.train()
        T,alpha,beta=4.0,0.5,0.1
        for X_b,y_b in g_train():
            with torch.no_grad(): z_t,f_t=teacher(X_b)
            z_s,f_s=student(X_b); fp=reg2(f_s)
            if method=="vanilla":
                lt=torch.stack([-z_t,z_t],1)/T; ls=torch.stack([-z_s,z_s],1)/T
                L=alpha*bce2(z_s,y_b)+(1-alpha)*(T**2)*F.kl_div(F.log_softmax(ls,1),F.softmax(lt,1).detach(),reduction="batchmean")
            elif method=="fitnet": L=bce2(z_s,y_b)+beta*F.mse_loss(fp,f_t.detach())
            else:
                ftn=F.normalize(f_t.detach(),dim=1); fsn=F.normalize(fp,dim=1)
                L=bce2(z_s,y_b)+beta*F.mse_loss(fsn,ftn)
            opt2.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(list(student.parameters())+list(reg2.parameters()),1.0); opt2.step()
        student.eval(); sc2,lb2=[],[]
        with torch.no_grad():
            for X_b,y_b in g_val():
                z_s,_=student(X_b); sc2.append(torch.sigmoid(z_s).cpu().numpy()); lb2.append(y_b.cpu().numpy())
        val_auc2=auroc(np.concatenate(lb2),np.concatenate(sc2))
        sch2.step(val_auc2)
        log(f"  Epoch {epoch:3d} | val AUROC {val_auc2:.4f}")
        if val_auc2>best_auc2+1e-5: best_auc2=val_auc2; best_state2={k:v.cpu().clone() for k,v in student.state_dict().items()}; pat2=0
        else:
            pat2+=1
            if pat2>=10: log(f"  Early stop @ {epoch}"); break
    student.load_state_dict(best_state2); student.to(device); student.eval()
    sc3,lb3=[],[]
    with torch.no_grad():
        for X_b,y_b in g_test():
            z_s,_=student(X_b); sc3.append(torch.sigmoid(z_s).cpu().numpy()); lb3.append(y_b.cpu().numpy())
    y_true_t=np.concatenate(lb3); y_score_t=np.concatenate(sc3)
    dual=dual_threshold_eval(y_true_t,y_score_t,cfg.EVAL_SENSITIVITY_TARGET)
    auc_pt,auc_lo,auc_hi=bootstrap_auroc_ci(y_true_t,y_score_t,n_resamples=500)
    lat=measure_latency_cpu(student.cpu(),n_features=n_features)
    log(f"  Test AUROC: {auc_pt:.4f} [{auc_lo:.4f}-{auc_hi:.4f}]")
    all_kd.append((method.upper(),auc_pt,y_true_t,y_score_t))
    kd_rows.append({"method":method.upper(),"student":f"S{gwo_s}","test_auroc":round(auc_pt,4),
        "test_auroc_ci":f"[{auc_lo:.4f}-{auc_hi:.4f}]","test_auprc":round(dual["auprc"],4),
        "latency_ms_cpu":round(lat,2)})

pd.DataFrame(kd_rows).to_csv(cfg.RESULTS_DIR/"kd_baselines.csv",index=False)
log("✅ kd_baselines.csv saved")

# DeLong tests
log("\n[DeLong: GWO-KD vs KD baselines]")
gwo_ckpt=torch.load(cfg.CHECKPOINTS_DIR/f"student_gwo_kd_s{gwo_s}.pt",map_location="cpu",weights_only=False)
gwo_st=build_student(gwo_s,n_features=n_features); gwo_st.load_state_dict(gwo_ckpt["state_dict"]); gwo_st.eval()
gsc,glb=[],[]
with torch.no_grad():
    for X_b,y_b in g_test():
        z,_=gwo_st(X_b.cpu()); gsc.append(torch.sigmoid(z).numpy()); glb.append(y_b.cpu().numpy())
y_gwo=np.concatenate(gsc); y_gtrue=np.concatenate(glb)
dl_rows=[]
for mname,tauc,ytrue,yscore in all_kd:
    dt=delong_test(y_gtrue,y_gwo,yscore)
    log(f"  GWO-KD vs {mname}: z={dt['z_stat']:.3f} p={dt['p_value']:.4f} {'✓' if dt['significant'] else '✗'}")
    dl_rows.append({"comparison":f"GWO-KD vs {mname}","gwo_auroc":round(auroc(y_gtrue,y_gwo),4),
        "other_auroc":round(tauc,4),"z_stat":round(dt['z_stat'],3),"p_value":round(dt['p_value'],4),"significant":dt['significant']})
pd.DataFrame(dl_rows).to_csv(cfg.RESULTS_DIR/"delong_kd_baselines.csv",index=False)
log("✅ delong_kd_baselines.csv saved")

log("\n[Stage 07 RECOVERY COMPLETE] ✓")
log("  → Run next: python src/08_sota_baselines.py")
