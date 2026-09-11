"""Render write-up figures from frozen records. No fitting or model calls."""

from pathlib import Path
import hashlib
import json
import random
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/presentation"
MET = ROOT / "results/metrics"
ORDER = ["bench_eval", "bench_deploy", "casual_eval", "casual_deploy"]
LABEL = dict(zip(ORDER, ["Benchmark eval", "Benchmark deploy", "Casual eval", "Casual deploy"]))
COLOUR = dict(zip(ORDER, ["#B65B23", "#246E92", "#865CA6", "#167B73"]))
INK = "#233140"
GREY = "#65717C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": GREY,
    "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#CDD3D8", "axes.titleweight": "normal",
    "svg.fonttype": "none", "savefig.facecolor": "white"})
EVIDENCE = {}


def save(fig, stem):
    for suffix in ["png", "svg"]:
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def heading(fig, title, subtitle):
    fig.text(.06, .955, title, fontsize=19, va="top")
    fig.text(.06, .910, subtitle, fontsize=11, color=GREY, va="top")


def dataset_figure(records):
    rng = random.Random(42)
    chosen = {}
    for q in ORDER:
        eligible = sorted((r for r in records if r["quadrant"] == q and r["split"] == "test"),
                          key=lambda r: r["stable_id"])
        chosen[q] = rng.choice(eligible)
    # User-requested compact replacement: retain the other three examples.
    chosen["bench_deploy"] = next(r for r in records
        if r["stable_id"] == "bench_deploy__672__51efeac3b04d" and r["split"] == "test")
    EVIDENCE["example_selection"] = {"seed": 42,
        "benchmark_deploy_selection": "Manually selected for brevity and three-line display",
        "benchmark_deploy_characters": len(chosen["bench_deploy"]["model_facing_text"])}
    EVIDENCE["random_examples"] = {q: {k: chosen[q][k] for k in ["stable_id", "model_facing_text"]} for q in ORDER}
    source = {"bench_eval": "MMLU / ARC / GSM8K", "bench_deploy": "ShareGPT, rewritten as benchmark questions",
              "casual_eval": "MuTual dialogue fragments", "casual_deploy": "Original ShareGPT user requests"}
    fig = plt.figure(figsize=(13, 4.9))
    # Remove the title area while preserving the card sizes and spacing.
    scale_y = 5.8 / 4.9
    fig.text(.06, .79 * scale_y, "Evaluation-labelled", fontsize=14)
    fig.text(.535, .79 * scale_y, "Deployment-labelled", fontsize=14)
    for q, (left, bottom) in zip(ORDER, [(0.06,.455),(.535,.455),(.06,.145),(.535,.145)]):
        height = .28
        ax = fig.add_axes([left, bottom * scale_y, .405, height * scale_y])
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.018",
                                   transform=ax.transAxes, facecolor="#F7F9FA", edgecolor="#E0E5E9"))
        ax.plot([.035,.965],[.985,.985],color=COLOUR[q],lw=3,transform=ax.transAxes)
        ax.text(.04,.88,LABEL[q],fontsize=13,color=COLOUR[q],va="top",transform=ax.transAxes)
        ax.text(.04,.70,source[q],fontsize=10,color=GREY,va="top",transform=ax.transAxes)
        text = "\n".join(textwrap.fill(line, width=66, replace_whitespace=False)
                         for line in chosen[q]["model_facing_text"].split("\n"))
        ax.text(.04,.49,text,fontsize=10.5,linespacing=1.42,va="top",transform=ax.transAxes)
    fig.text(.06,.075 * scale_y,"Each group: 300 prompts · 210 training / 90 test",fontsize=11)
    fig.text(.06,.032 * scale_y,"Labels come from dataset source; they do not establish what the model believes about its setting.",
             fontsize=10.5,color=GREY)
    save(fig,"01_dataset_examples")


def english_figure():
    curve = pd.read_csv(MET / "probe_b_layer_cv.csv").sort_values("layer_number")
    folds = pd.read_csv(MET / "probe_b_cv_folds.csv")
    computed = folds.groupby("layer_number").validation_auroc.mean()
    np.testing.assert_allclose(computed.values,curve.mean_cv_auroc.values)
    assert curve.layer_number.tolist() == list(range(1,33))
    pooled = pd.read_csv(MET / "tfidf_probe_b_cv.csv").validation_auroc.to_numpy()
    within = pd.read_csv(MET / "tfidf_frozen_format_cv.csv")
    baseline = [pooled, within[within["format"] == "benchmark"].validation_auroc.to_numpy(),
                within[within["format"] == "casual"].validation_auroc.to_numpy()]
    EVIDENCE["english"] = {"block_1_cv":float(curve.iloc[0].mean_cv_auroc),
                            "block_32_cv":float(curve.iloc[-1].mean_cv_auroc),
                            "tfidf_means":dict(zip(["pooled","benchmark","casual"],map(lambda x:float(x.mean()),baseline)))}
    fig = plt.figure(figsize=(13,6.7))
    heading(fig,"The English labels are easy to recover early",
            "Qwen3.5-9B activation probes and a separate classifier using only the words in each prompt.")
    ax = fig.add_axes([.07,.25,.30,.51])
    ax.set_title("A   Probe B by block",loc="left",pad=22,fontsize=13)
    ax.fill_between(curve.layer_number,curve.min_cv_auroc,curve.max_cv_auroc,color="#246E92",alpha=.10,lw=0)
    ax.plot(curve.layer_number,curve.mean_cv_auroc,color="#246E92",lw=2.3)
    ax.scatter([1,32],[curve.iloc[0].mean_cv_auroc,curve.iloc[-1].mean_cv_auroc],color="#246E92",s=40,zorder=4,clip_on=False)
    ax.annotate("Block 1: 0.991",(1,curve.iloc[0].mean_cv_auroc),xytext=(2,.91),
                arrowprops={"arrowstyle":"-","color":GREY},fontsize=11)
    ax.annotate("Block 32: 0.997",(32,curve.iloc[-1].mean_cv_auroc),xytext=(17,.84),
                arrowprops={"arrowstyle":"-","color":GREY},fontsize=11)
    ax.set(xlim=(.5,32.5),ylim=(.5,1.0),xlabel="Decoder block",ylabel="Mean CV AUROC")
    ax.set_xticks([1,8,16,24,32]);ax.set_yticks([.5,.6,.7,.8,.9,1.0])
    ax.grid(axis="y",alpha=.20);ax.set_axisbelow(True)
    ax = fig.add_axes([.56,.25,.39,.51])
    ax.set_title("B   Words alone (TF–IDF)",loc="left",pad=22,fontsize=13)
    for y,values in zip([2,1,0],baseline):
        ax.scatter(values,y+np.linspace(-.07,.07,len(values)),s=27,color="#ADBAC6",zorder=3)
        ax.scatter(values.mean(),y,s=70,color=INK,marker="D",zorder=4)
        ax.text(values.mean()+.018,y+.13,f"{values.mean():.3f}",fontsize=11)
    ax.set_yticks([2,1,0],["All groups","Benchmark only","Casual only"])
    ax.set(xlim=(.5,1.015),ylim=(-.55,2.55),xlabel="CV AUROC")
    ax.set_xticks([.5,.6,.7,.8,.9,1]);ax.axvline(.5,color=GREY,lw=1,ls=":")
    ax.grid(axis="x",alpha=.20);ax.set_axisbelow(True);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    fig.text(.07,.15,"All results: five-fold CV on training data. Each block has its own fitted probe.",fontsize=10.5)
    fig.text(.07,.108,"A: line = fold mean; shading = fold range, not a confidence interval. B: dots = folds; diamonds = means.",fontsize=10,color=GREY)
    fig.text(.07,.066,"Early separability and word-only success do not establish which features the activation probe uses.",fontsize=10.5,color=GREY)
    save(fig,"02_english_layers_and_baselines")


def translation_figure(records):
    d = pd.read_csv(MET / "stage9_multilingual_predictions.csv")
    test_ids = {r["stable_id"] for r in records if r["split"]=="test"}
    stats = {}
    fig = plt.figure(figsize=(13,7.8))
    heading(fig,"Translation changes both scores and separation",
            "The same 360 held-out prompts; the English Probe B is applied without refitting or recalibration.")
    for i,language in enumerate(["english","spanish","japanese"]):
        sub = d[d.language==language].copy()
        assert len(sub)==360 and set(sub.stable_id)==test_ids and not sub.stable_id.duplicated().any()
        assert sub.groupby("quadrant").size().to_dict()=={q:90 for q in ORDER}
        np.testing.assert_array_equal((sub.decision_score>=0).astype(int),sub.predicted_class)
        scores = {"pooled_auroc":roc_auc_score(sub.context_label,sub.decision_score),
                  "accuracy":float((sub.predicted_class==sub.context_label).mean())}
        for f in ["bench","casual"]:
            w=sub[sub.quadrant.str.startswith(f)]
            scores[f+"_auroc"]=roc_auc_score(w.context_label,w.decision_score)
        stats[language]=scores
        left=.16+i*.265
        ax=fig.add_axes([left,.36,.235,.40])
        ax.set_title(language.title(),fontsize=14,loc="left",pad=25)
        rng=np.random.default_rng(42)
        for y,q in zip([3,2,1,0],ORDER):
            values=sub[sub.quadrant==q].sort_values("stable_id").decision_score.to_numpy()
            ax.scatter(values,y+rng.uniform(-.18,.18,len(values)),s=10,alpha=.42,color=COLOUR[q],edgecolors="none",zorder=2)
            lo,med,hi=np.percentile(values,[25,50,75])
            ax.plot([lo,hi],[y,y],color=COLOUR[q],lw=5,solid_capstyle="round",zorder=3)
            ax.scatter([med],[y],marker="|",s=130,color=INK,zorder=4)
        ax.axvline(0,color=INK,ls="--",lw=1.1,zorder=1)
        ax.text(0,3.63,"English threshold",ha="center",fontsize=8.5,color=GREY)
        ax.set(xlim=(-35,25),ylim=(-.5,3.95),xlabel="Probe score")
        ax.set_xticks([-30,-15,0,15]);ax.set_yticks([3,2,1,0], [LABEL[q] for q in ORDER] if i==0 else [])
        if i==0:
            for label,q in zip(ax.get_yticklabels(),ORDER):label.set_color(COLOUR[q])
        ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False);ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
        for y,k,fmt in [(.26,"pooled_auroc",".3f"),(.22,"bench_auroc",".3f"),(.18,"casual_auroc",".3f"),(.13,"accuracy",".1%")]:
            fig.text(left+.115,y,format(scores[k],fmt),ha="center",fontsize=12,color=INK)
    for y,label in [(.26,"Pooled AUROC"),(.22,"Benchmark AUROC"),(.18,"Casual AUROC"),(.13,"Threshold accuracy")]:
        fig.text(.022,y,label,fontsize=10.5)
    fig.text(.06,.085,"Dots = all prompts; thick lines = middle 50%; marks = medians. All panels use the same score axis.",fontsize=10,color=GREY)
    fig.text(.06,.05,"A score shift can break the threshold without changing AUROC; the benchmark AUROC drop also shows weaker ranking.",fontsize=10,color=GREY)
    EVIDENCE["translation"]=stats
    save(fig,"03_translation_scores")


def baselines_figure():
    prediction_dir = ROOT / 'results/predictions'
    def test_auc(name):
        frame = pd.read_csv(prediction_dir / name)
        assert len(frame) == 360 and set(frame.split) == {'test'}
        return float(roc_auc_score(frame.context_label, frame.decision_score))
    activation = test_auc('probe_b_english.csv')
    length = test_auc('probe_b_length_english.csv')
    shuffled = pd.read_csv(prediction_dir / 'probe_b_shuffled_english.csv')
    shuffle_aucs = np.array([roc_auc_score(g.context_label, g.decision_score)
                            for _, g in shuffled.groupby('shuffle_seed')])
    np.testing.assert_allclose(shuffle_aucs, pd.read_csv(MET / 'probe_b_shuffled_baseline.csv').sort_values('shuffle_seed').auroc)
    pooled = pd.read_csv(MET / 'tfidf_probe_b_cv.csv').validation_auroc.mean()
    formats = pd.read_csv(MET / 'tfidf_frozen_format_cv.csv').groupby('format').validation_auroc.mean()
    values = [activation, pooled, formats['benchmark'], formats['casual'], length, shuffle_aucs.mean()]
    samples = pd.read_csv(MET / 'tfidf_selection_sensitivity_seeds.csv').sort_values('seed')
    folds = pd.read_csv(MET / 'tfidf_selection_sensitivity_folds.csv')
    np.testing.assert_allclose(samples.mean_cv_auroc, folds.groupby('seed').validation_auroc.mean())
    assert len(samples) == 20 and (samples.overlap_with_frozen_test == 0).all()
    unmatched = samples.mean_cv_auroc.mean()

    fig = plt.figure(figsize=(13, 6.7))
    heading(fig, 'Raw prompt text predicts labels within each format',
            'Simple baselines identify alternatives to evaluation awareness; word-only success persists without pairwise matching.')
    ax = fig.add_axes([.07, .27, .49, .48])
    ax.set_title('A   Activation, text and control baselines', loc='left', pad=22, fontsize=13)
    colours = ['#246E92', '#167B73', '#167B73', '#167B73', '#65717C', '#ADBAC6']
    for x, (value, colour) in enumerate(zip(values, colours)):
        ax.scatter(x, value, color=colour, s=85, marker='D' if x in [1, 2, 3] else 'o', zorder=4)
        ax.text(x, value + .023, f'{value:.3f}', ha='center', fontsize=11)
    ax.axhline(.5, color=GREY, ls='--', lw=1)
    ax.text(-.45, .505, 'Chance', ha='left', va='bottom', color=GREY, fontsize=9)
    ax.set(xlim=(-.55, 5.55), ylim=(.45, 1.075), ylabel='AUROC')
    ax.set_xticks(range(6), ['Activation\nprobe', 'TF–IDF\npooled', 'TF–IDF\nbenchmark',
                          'TF–IDF\ncasual', 'Length\nonly', 'Shuffled\nlabels'], fontsize=10)
    ax.set_yticks([.5, .6, .7, .8, .9, 1])
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax = fig.add_axes([.69, .27, .26, .48])
    ax.set_title('B   Success without matching', loc='left', pad=22, fontsize=13)
    jitter = np.random.default_rng(42).uniform(-.13, .13, len(samples))
    ax.scatter(jitter, samples.mean_cv_auroc, color='#B65B23', alpha=.72, s=38, zorder=3)
    ax.axhline(pooled, color='#246E92', lw=2, label=f'Matched v2: {pooled:.3f}')
    ax.axhline(unmatched, color='#B65B23', lw=1.8, ls='--', label=f'Unmatched mean: {unmatched:.3f}')
    ax.set(xlim=(-.5, .5), ylim=(.88, .946), ylabel='Mean training CV AUROC')
    ax.set_yticks([.88, .89, .90, .91, .92, .93, .94])
    ax.set_xticks([0], ['20 unmatched resamples'], fontsize=10)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.07, .17, 'A: circles = held-out test; diamonds = five-fold training CV. The estimates use different evaluation sets.', fontsize=10.5)
    fig.text(.07, .12, f'Activation: Probe B, block 32. Length: characters + Qwen tokens. Shuffled labels: mean of {len(shuffle_aucs)} fits.', fontsize=10, color=GREY)
    fig.text(.07, .07, 'B: each dot is a five-fold CV mean on 840 prompts, sampled within length bounds but without pairwise matching.', fontsize=10, color=GREY)
    save(fig, '04_baselines_and_matching')
    EVIDENCE['baselines_and_matching'] = {
        'activation_test_auroc': activation, 'pooled_tfidf_cv': float(pooled),
        'benchmark_tfidf_cv': float(formats['benchmark']), 'casual_tfidf_cv': float(formats['casual']),
        'length_test_auroc': length, 'shuffled_mean_test_auroc': float(shuffle_aucs.mean()),
        'unmatched_mean_cv': float(unmatched), 'unmatched_seed_cv': samples[['seed', 'mean_cv_auroc']].to_dict('records')}
    inputs = [prediction_dir / p for p in ['probe_b_english.csv', 'probe_b_length_english.csv', 'probe_b_shuffled_english.csv']]
    inputs += [MET / p for p in ['probe_b_shuffled_baseline.csv', 'tfidf_probe_b_cv.csv', 'tfidf_frozen_format_cv.csv',
                                'tfidf_selection_sensitivity_seeds.csv', 'tfidf_selection_sensitivity_folds.csv']]
    EVIDENCE['baselines_and_matching_input_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}


def dataset_design_figure():
    from src.dataset_design_diagram import draw_dataset_design
    save(draw_dataset_design(), '05_dataset_design')


def source_overlap_figure(records):
    subsets = {q: [r for r in records if r['quadrant'] == q]
               for q in ['bench_deploy', 'casual_deploy']}
    maps = {q: {r['source_id']: r for r in rows} for q, rows in subsets.items()}
    for q in maps:
        assert len(maps[q]) == len(subsets[q]) and all(maps[q])
    links = []
    for source_id in sorted(maps['bench_deploy'].keys() & maps['casual_deploy'].keys()):
        rewritten, original = maps['bench_deploy'][source_id], maps['casual_deploy'][source_id]
        links.append({'source_id': source_id, 'rewritten_stable_id': rewritten['stable_id'],
                      'original_stable_id': original['stable_id'], 'rewritten_split': rewritten['split'],
                      'original_split': original['split'],
                      'crosses_train_test': rewritten['split'] != original['split']})
    linked = pd.DataFrame(links)
    linked.to_csv(OUT / '06_shared_source_links.csv', index=False)
    pairs = [('train', 'train'), ('train', 'test'), ('test', 'train'), ('test', 'test')]
    counts = [int(((linked.rewritten_split == a) & (linked.original_split == b)).sum()) for a, b in pairs]
    cross_count = int(linked.crosses_train_test.sum())
    assert counts == [19, 7, 14, 3]
    fig = plt.figure(figsize=(13, 5.8))
    fig.text(.06, .955, 'Some source prompts appear in both training and test', fontsize=19, va='top')
    fig.text(.06, .885, 'Shared ShareGPT source IDs · arrows show rewritten version → original version',
             fontsize=11, color=GREY, va='top')
    ax = fig.add_axes([.18, .25, .75, .52])
    ax.barh([3, 2, 1, 0], counts, height=.55,
            color=['#246E92', '#B65B23', '#B65B23', '#ADBAC6'], zorder=3)
    for y, count in zip([3, 2, 1, 0], counts):
        ax.text(count + .3, y, str(count), fontsize=12, va='center')
    ax.set_yticks([3, 2, 1, 0], ['Train → train', 'Train → test', 'Test → train', 'Test → test'])
    ax.set(xlim=(0, 21), ylim=(-.55, 3.55), xlabel='Shared source IDs')
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.06, .12, f'{len(linked)} shared source IDs in total; {cross_count} cross the training/test boundary (orange bars).', fontsize=11)
    fig.text(.06, .065, 'Shared IDs link related versions of a prompt, not necessarily identical text.', fontsize=10.5, color=GREY)
    save(fig, '06_source_overlap')
    EVIDENCE['source_overlap'] = {'direction': 'rewritten (bench_deploy) → original (casual_deploy)',
        'counts': {f'{a}_to_{b}': count for (a, b), count in zip(pairs, counts)},
        'shared_source_ids': len(linked), 'cross_train_test_source_ids': cross_count,
        'dataset_sha256': hashlib.sha256((ROOT / 'data/selected/english_selected_v2.jsonl').read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/"data/selected/english_selected_v2.jsonl"
    manifest=json.loads((ROOT/"data/selected/english_selected_v2_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["selected_data_sha256"]
    records=[json.loads(line) for line in path.read_text().splitlines()]
    dataset_figure(records);english_figure();translation_figure(records);baselines_figure();dataset_design_figure();source_overlap_figure(records)
    inputs=[path,MET/"probe_b_layer_cv.csv",MET/"probe_b_cv_folds.csv",MET/"tfidf_probe_b_cv.csv",
            MET/"tfidf_frozen_format_cv.csv",MET/"stage9_multilingual_predictions.csv"]
    EVIDENCE["input_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/"figure_data.json").write_text(json.dumps(EVIDENCE,indent=2)+"\n")
    print(json.dumps(EVIDENCE["translation"],indent=2))
    print(f"Saved six PNG and six SVG figures to {OUT}")


if __name__=="__main__":
    main()
