# Tree Ensembles

[Feature Engineering for ML](01-feature-engineering-for-ml.md) closed by freezing `data/part7.parquet` and naming the budget it contains: 16,592 rows worth roughly 1,750 independent observations, a 0.515 honest AUC from a linear probe, and a base rate that hands any model 58% accuracy for free. This lesson spends that budget on the models with the strongest claim to it. On tabular data — rows of heterogeneous, weakly related columns, exactly what the feature matrix is — tree ensembles have dominated applied machine learning for a decade: they need no scaling, ignore monotone transforms, capture interactions without being told where to look, and shrug at the correlated features that made lesson one nervous.

The lesson is structured as a controlled experiment, and the order of sections is the experiment's design. The ensembles go first and get everything they ask for: five hundred trees, two gradient-boosting libraries, a hyperparameter search under cross-validation that respects the `t1` column. Only then does the baseline appear — a logistic regression that trains in under a second — followed by the two skills that outlive any particular model: reading feature importance without being lied to, and turning scores into probabilities you could actually size a position with. Every split in this lesson is chronological, every fit passes the uniqueness weights `w` as `sample_weight`, and every number is pinned from the frozen file.

## Why trees fit tabular finance — and where they stop

A decision tree is a piecewise-constant function grown greedily: find the split that most purifies the labels, recurse. That bias suits financial features — thresholds ("volatility above its 80th percentile") are how discretionary traders already think, and no one has to guess whether the relationship is linear. The catch is capacity. An unrestricted tree can carve the feature space until every leaf holds a single training row, which on a low signal-to-noise dataset is a memorization machine. The first exhibit trains both extremes on SPY — training data through 2016, test from 2017, a chronological split whose one sin (labels born in late December 2016 resolve into January) is a single label horizon wide; section four repairs it properly with purging.

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

mat = pd.read_parquet("data/part7.parquet")
spy = mat[mat.symbol == "SPY"]
feat = [k for k in spy.columns if k.startswith("f_")]
X, y = spy[feat], (spy.y_tb > 0).astype(int)
tr, te = spy.index <= "2016", spy.index >= "2017"
print(f"the frozen matrix: {len(mat):,} rows; SPY slice {len(spy):,}, "
      f"train {tr.sum():,} (to 2016), test {te.sum():,} (2017+), base rate {y[te].mean():.3f}")

deep = DecisionTreeClassifier(random_state=0).fit(X[tr], y[tr])
stump = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[tr], y[tr])
print(f"unlimited tree: {deep.get_n_leaves():,} leaves, "
      f"train acc {deep.score(X[tr], y[tr]):.3f}, test acc {deep.score(X[te], y[te]):.3f}")
print(f"depth-3 tree:   {stump.get_n_leaves()} leaves,    "
      f"train acc {stump.score(X[tr], y[tr]):.3f}, test acc {stump.score(X[te], y[te]):.3f}")
print(f"depth-3 root split: {feat[stump.tree_.feature[0]]} <= {stump.tree_.threshold[0]:.4f}")
# => the frozen matrix: 16,592 rows; SPY slice 6,154, train 3,773 (to 2016), test 2,129 (2017+), base rate 0.632
#    unlimited tree: 475 leaves, train acc 1.000, test acc 0.490
#    depth-3 tree:   8 leaves,    train acc 0.619, test acc 0.606
#    depth-3 root split: f_amihud <= 0.0432
```

The unrestricted tree is the purest overfitting exhibit this course will ever print: 475 leaves, a *perfect* 1.000 on the years it saw, 0.490 — worse than a coin — on the years it did not. It learned the biography of 2001–2016, not the physics of markets, and [Part IV's validation lesson](../part-04-strategy-development/08-validation-and-overfitting.md) named this exact failure at the strategy level. The depth-3 tree, eight leaves, tells the more interesting story. Its 0.606 test accuracy looks respectable until you read it against the base rate printed above it: on the 2017+ window, "always up" scores 0.632, so the honest little tree also *loses to doing nothing* — a reminder that in a drifting market, accuracy is a rigged metric, which is why AUC (which the base rate cannot inflate) does the grading from here on. And note what the root split chose: not momentum but `f_amihud`, the illiquidity proxy — the single most label-relevant threshold in the training years was "is liquidity normal or strained," a fact worth remembering when the importance rankings return in section six.

## A forest averages away variance, not noise

The random forest fixes the unrestricted tree's disease with two doses of randomness: each tree sees a bootstrap resample of the rows, and each split considers only a random subset of features. Five hundred such trees disagree with each other by construction, and averaging their votes cancels the idiosyncratic carving — variance reduction, the textbook cure from the [bias–variance decomposition](../appendix/part-14-model-selection/01-bias-variance-tradeoff.md). The open question is whether variance was ever the binding constraint:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
X, y, w = spy[feat], (spy.y_tb > 0).astype(int), spy.w
tr, te = spy.index <= "2016", spy.index >= "2017"

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=50, random_state=42,
                            n_jobs=4).fit(X[tr], y[tr], sample_weight=w[tr])
per_tree = np.stack([t.predict_proba(X[te].values)[:, 1] for t in rf.estimators_])
tree_aucs = [roc_auc_score(y[te], p) for p in per_tree]
print(f"single trees: mean AUC {np.mean(tree_aucs):.3f} "
      f"(range {min(tree_aucs):.3f}-{max(tree_aucs):.3f})")
print(f"forest of 500: AUC {roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]):.3f}")
print(f"per-day spread across trees: mean std {per_tree.std(axis=0).mean():.3f}; "
      f"forest prediction std {rf.predict_proba(X[te])[:, 1].std():.3f}")
# => single trees: mean AUC 0.497 (range 0.423-0.554)
#    forest of 500: AUC 0.492
#    per-day spread across trees: mean std 0.198; forest prediction std 0.075
```

The mechanism worked flawlessly and the outcome is null — hold both halves of that sentence. On any given test day the five hundred trees disagree with a standard deviation of 0.198, nearly twenty probability points, and the averaged forecast compresses to a docile 0.075 spread around the base rate: variance reduction did exactly what the textbook promises. But the forest's AUC is 0.492, statistically indistinguishable from — actually a hair *below* — the 0.497 average of its own constituents, whose individual AUCs range from 0.423 to 0.554, a spread that is itself just sampling noise around one half. Averaging five hundred opinions helps when the opinions share a signal and differ by noise; here they differ by noise and share *nothing*, so the average is a smoother version of the same coin flip. This is the lesson's first encounter with a theme that will recur through the deep-learning scoreboard: in this domain the constraint is not estimation variance, which more trees, more data, or more averaging can fix, but the information content of the features — and no amount of ensemble machinery manufactures information that is not there. The `min_samples_leaf=50` floor, note, already protects each tree from section one's 475-leaf fate; the forest's problem is not overfitting. It has nothing left to overfit *to*.

## Boosting meets an honest referee

Gradient boosting grows trees sequentially, each fitted to the residual errors of the ensemble so far — a fundamentally more aggressive posture than the forest's independent averaging, and the engine behind XGBoost and LightGBM, the two libraries that have won essentially every tabular ML competition of the last decade. Aggression needs a stopping rule. The standard discipline is early stopping: hold out a validation slice, add trees only while validation loss improves, and stop when it turns. Our validation tail is embargoed — it starts 21 trading days after the training data ends, one label horizon, so no label spans the boundary:

```python
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
X, y, w = spy[feat], (spy.y_tb > 0).astype(int), spy.w
tr = spy.index <= "2014"
va = (spy.index >= "2015-02") & (spy.index <= "2016-12")   # embargoed validation tail
te = spy.index >= "2017"

xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4,
                    tree_method="hist", nthread=4, seed=42,
                    early_stopping_rounds=100, eval_metric="logloss")
xgb.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])], verbose=False)
gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=15,
                         seed=42, deterministic=True, force_row_wise=True,
                         num_threads=1, verbosity=-1)
gbm.fit(X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])],
        callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"with an embargoed referee: xgboost stops at round {xgb.best_iteration}, "
      f"lightgbm at round {gbm.best_iteration_} (of 2000 offered)")

for name, m in {
    "xgboost ": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              tree_method="hist", nthread=4, seed=42),
    "lightgbm": lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                   seed=42, deterministic=True, force_row_wise=True,
                                   num_threads=1, verbosity=-1),
}.items():
    t0 = time.perf_counter()
    m.fit(X[tr], y[tr], sample_weight=w[tr])
    dt = time.perf_counter() - t0
    print(f"{name} forced to 300 rounds: train AUC "
          f"{roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1]):.3f}, "
          f"test AUC {roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]):.3f}  ({dt:.1f}s)")
# => with an embargoed referee: xgboost stops at round 0, lightgbm at round 1 (of 2000 offered)
#    xgboost  forced to 300 rounds: train AUC 0.946, test AUC 0.477  (0.1s)
#    lightgbm forced to 300 rounds: train AUC 0.992, test AUC 0.467  (0.1s)
```

The first line deserves to be framed. Offered two thousand rounds of boosting and an honest, embargoed referee, XGBoost stopped at round *zero* and LightGBM at round *one*: from the very first tree, every additional unit of fitting made validation loss worse, so the machinery — working exactly as designed — concluded that the best available model is approximately the prior. This is not the libraries failing; it is the libraries being the only participant in the experiment with no ego. The second and third lines show what overriding the referee buys. Forced to 300 rounds, the boosters do what boosting does — train AUC 0.946 and 0.992, a tenth of a second each — and deliver 0.477 and 0.467 out of sample, *below* one half, having spent 300 rounds annotating the residual noise of 2001–2014 in exquisite detail. On Kaggle-style tabular problems these same settings and these same libraries genuinely win, because those datasets hide learnable structure. The pair of printouts is the cleanest statement so far of what makes financial daily bars different: the structure that exists ([lesson one](01-feature-engineering-for-ml.md) measured it at AUC 0.515) is thinner than one boosting round's appetite.

## Tuning without self-deception

The standard rebuttal to section three is "you used the wrong hyperparameters," and the standard workflow answers it with a grid search under cross-validation. Part IV's validation lesson showed that *shuffled* folds manufacture skill (+0.061 from nothing) on overlapping labels; the repair is purged, embargoed, contiguous folds — and now that the dataset carries an explicit `t1` column, the purge can finally be exact: drop every training row whose label was still unresolved when the test fold began. The search runs on data through 2021, and 2022+ stays untouched until the grid has committed:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
dev, hold = spy[spy.index <= "2021"], spy[spy.index >= "2022"]

def purged_folds(t1, n_splits=5, embargo=21):    # fold geometry of Part IV, lesson eight
    idx, n = t1.index, len(t1)
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        t_max = t1.iloc[a:b].max()
        train = [i for i in range(n) if (i < a and t1.iloc[i] < idx[a])
                 or (i >= b + embargo and idx[i] > t_max)]
        yield np.array(train), np.arange(a, b)

def cv_auc(leaves, lr):
    aucs = []
    for tr, te in purged_folds(dev.t1):
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=leaves, learning_rate=lr,
                               seed=42, deterministic=True, force_row_wise=True,
                               num_threads=1, verbosity=-1)
        m.fit(dev[feat].iloc[tr], (dev.y_tb.iloc[tr] > 0), sample_weight=dev.w.iloc[tr])
        aucs.append(roc_auc_score((dev.y_tb.iloc[te] > 0), m.predict_proba(dev[feat].iloc[te])[:, 1]))
    return np.mean(aucs)

grid = {(lv, lr): cv_auc(lv, lr) for lv in [7, 15, 31] for lr in [0.02, 0.05, 0.10]}
ranked = sorted(grid, key=grid.get, reverse=True)
best, median = ranked[0], ranked[4]
print(f"purged 5-fold CV over 9 configs: best {best} AUC {grid[best]:.3f}, "
      f"median {median} AUC {grid[median]:.3f}, worst AUC {grid[ranked[-1]]:.3f}")

def hold_auc(leaves, lr):
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=leaves, learning_rate=lr,
                           seed=42, deterministic=True, force_row_wise=True,
                           num_threads=1, verbosity=-1)
    m.fit(dev[feat], (dev.y_tb > 0), sample_weight=dev.w)
    return roc_auc_score((hold.y_tb > 0), m.predict_proba(hold[feat])[:, 1])

print(f"untouched 2022+ holdout: best-by-CV {hold_auc(*best):.3f}, "
      f"median config {hold_auc(*median):.3f}")
# => purged 5-fold CV over 9 configs: best (31, 0.05) AUC 0.462, median (31, 0.1) AUC 0.455, worst AUC 0.440
#    untouched 2022+ holdout: best-by-CV 0.398, median config 0.414
```

Read the first line's spread before its ranking: nine configurations landed between 0.440 and 0.462 — twenty-two thousandths of AUC separating "best" from "worst," on folds whose sampling noise is at least that large. The grid is not measuring hyperparameter quality; it is drawing lots. The second line is the punchline the setup guaranteed a hearing: on the untouched holdout, the CV champion scores 0.398 and the median configuration 0.414 — *the reward for winning the tournament is losing hardest afterward*, [Part IV's probability-of-backtest-overfitting](../part-04-strategy-development/08-validation-and-overfitting.md) logic recurring one level up, at hyperparameter scale. Two disciplines keep this result from being misread. First: every AUC on the board is below 0.5, and the tempting response — flip the model's sign — is itself a selection made after seeing the holdout, the exact crime this section exists to prevent; a sub-0.5 AUC in a near-zero-signal problem is noise, not inverted signal. Second: the holdout is 2022, a bear market unlike anything in the 2010s training data, and a model tuned on one regime meeting another is not an edge case — it is the [permanent operating condition](../part-03-statistics/06-bayesian-methods-and-hmms.md) that [Production ML](05-production-ml.md) will monitor for. When a real search over a real grid is warranted, [Bayesian Optimization](../advanced/01-bayesian-optimization.md) does it with fewer evaluations; nothing about a smarter searcher changes what is true here, which is that there is little to find.

## The bar: a logistic regression that took a second

Now the baseline — and the economics. Four models, identical purged training data, identical features, and then the only scoreboard that matters: each model's probabilities run as a long/flat SPY strategy, next-day execution, charged at [Part IV's cost model](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) of half-spread plus commission. The bar is public: tsmom's full-sample 0.30, buy-and-hold's 0.38, both recomputed on the same 2017+ window the models are graded on.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
X, y, w = spy[feat], (spy.y_tb > 0).astype(int), spy.w
tr, te = spy.index <= "2016", spy.index >= "2017"
c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
r = np.log(c).diff().reindex(spy.index[te])
COST = (0.5 + 0.2) * 1e-4                        # half-spread + commission, Part IV lesson seven

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "forest":   RandomForestClassifier(n_estimators=500, min_samples_leaf=50,
                                       random_state=42, n_jobs=4),
    "xgboost":  XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              tree_method="hist", nthread=4, seed=42),
    "lightgbm": lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                   seed=42, deterministic=True, force_row_wise=True,
                                   num_threads=1, verbosity=-1),
}
print("2017+, long/flat on P(up) > 0.5, costs 0.7 bp per one-way trade:")
for name, m in models.items():
    kw = {"logisticregression__sample_weight": w[tr]} if name == "logistic" else {"sample_weight": w[tr]}
    m.fit(X[tr], y[tr], **kw)
    p = pd.Series(m.predict_proba(X[te])[:, 1], index=spy.index[te])
    pos = (p > 0.5).astype(float)
    net = (pos.shift(1) * r - pos.diff().abs() * COST).dropna()
    print(f"  {name:8s}: AUC {roc_auc_score(y[te], p):.3f}   net Sharpe "
          f"{np.sqrt(252) * net.mean() / net.std():.2f}   days long {pos.mean():.0%}")

sig = np.sign(np.log(c).diff().rolling(252).sum())     # the Part IV rule, same window
ts = (sig.shift(1) * np.log(c).diff() - sig.diff().abs() * COST).reindex(spy.index[te]).dropna()
bh = np.log(c).diff().reindex(spy.index[te]).dropna()
print(f"  tsmom   : net Sharpe {np.sqrt(252) * ts.mean() / ts.std():.2f}   (0.30 full-sample)")
print(f"  buy-hold: net Sharpe {np.sqrt(252) * bh.mean() / bh.std():.2f}   (0.38 full-sample)")
# => 2017+, long/flat on P(up) > 0.5, costs 0.7 bp per one-way trade:
#      logistic: AUC 0.478   net Sharpe 0.38   days long 16%
#      forest  : AUC 0.492   net Sharpe 0.17   days long 38%
#      xgboost : AUC 0.471   net Sharpe -0.01   days long 14%
#      lightgbm: AUC 0.481   net Sharpe 0.20   days long 13%
#      tsmom   : net Sharpe 0.18   (0.30 full-sample)
#      buy-hold: net Sharpe 0.71   (0.38 full-sample)
```

The scoreboard reads top to bottom as an indictment in three counts. Count one: the logistic regression — one second of training, [a model from the appendix](../appendix/part-13-regression/04-logistic-regression.md) — posts the best net Sharpe of the four models, 0.38, with the forest at 0.17, LightGBM at 0.20, and XGBoost at −0.01. The ensembles' extra capacity bought nothing that survived costs. Count two: every model loses to buy-and-hold's 0.71, and so does tsmom's 0.18 — on a window that was mostly one long bull market, the winning move was the one with no model at all, which is why a scoreboard must always carry its window's context rather than pretend 2017–2025 is destiny. Count three, subtler: look at the exposure column. The models are long only 13–38% of days — trained on uniqueness-weighted labels whose effective sample is small, their probabilities hug the middle and rarely clear 0.5 with conviction, so they sat out most of a rally. That caution is not a bug; it is an honest reading of how little their features know, and the AUC column (0.471–0.492, all within noise of the coin) says the same thing in metric form. Note also that AUC rank and Sharpe rank disagree — the forest has the best AUC and a middling Sharpe — because a threshold, a lag, and a cost model stand between a score and a P&L; the trade, as always in this course, is graded at the exit, not at the forecast.

## Feature importance lies about correlated features

Even a model that cannot predict is interrogated about *what it looked at*, and here the standard tools mislead in a specific, mechanical way. Mean decrease in impurity (MDI) — the default `feature_importances_` — credits features by how much they purified training splits: an in-sample bookkeeping entry that never consults the test set. Permutation importance asks the out-of-sample question directly: shuffle one column and watch skill fall. And both are haunted by lesson one's correlation table, because when two features carry the same information, the model can take it from either — so credit splits arbitrarily, and deleting the "important" one costs nothing. Lesson one built the perfect test pair: close-to-close and Parkinson volatility, correlated at 0.97.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
X, y, w = spy[feat], (spy.y_tb > 0).astype(int), spy.w
tr, te = spy.index <= "2016", spy.index >= "2017"

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=50, random_state=42,
                            n_jobs=4).fit(X[tr], y[tr], sample_weight=w[tr])
mdi = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
perm = permutation_importance(rf, X[te], y[te], n_repeats=20, random_state=0,
                              scoring="roc_auc", n_jobs=4)
pi = pd.Series(perm.importances_mean, index=feat).sort_values(ascending=False)
print(f"MDI top-3 (in-sample structure): {list(mdi.index[:3])}")
print(f"permutation top-3 (out-of-sample skill): {list(pi.index[:3])}, "
      f"best {pi.iloc[0]:.4f} +/- {perm.importances_std[list(feat).index(pi.index[0])]:.4f}")

# the substitution effect, staged on the 0.97-correlated pair from lesson one
print(f"corr(f_vol_21, f_park) = {X.f_vol_21.corr(X.f_park):.2f}; "
      f"MDI shares: f_vol_21 {mdi['f_vol_21']:.3f}, f_park {mdi['f_park']:.3f}")
rf2 = RandomForestClassifier(n_estimators=500, min_samples_leaf=50, random_state=42,
                             n_jobs=4).fit(X[tr].drop(columns="f_vol_21"), y[tr],
                                           sample_weight=w[tr])
mdi2 = pd.Series(rf2.feature_importances_, index=[f for f in feat if f != "f_vol_21"])
a1 = roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1])
a2 = roc_auc_score(y[te], rf2.predict_proba(X[te].drop(columns="f_vol_21"))[:, 1])
print(f"delete f_vol_21: f_park's MDI {mdi['f_park']:.3f} -> {mdi2['f_park']:.3f}, "
      f"AUC {a1:.3f} -> {a2:.3f}")
# => MDI top-3 (in-sample structure): ['f_amihud', 'f_ret_252', 'f_volratio']
#    permutation top-3 (out-of-sample skill): ['f_amihud', 'f_park', 'f_ret_252'], best 0.0077 +/- 0.0054
#    corr(f_vol_21, f_park) = 0.97; MDI shares: f_vol_21 0.072, f_park 0.071
#    delete f_vol_21: f_park's MDI 0.072 -> 0.081, AUC 0.492 -> 0.497
```

Start with the number attached to the permutation "winner": 0.0077 of AUC, with a standard deviation of 0.0054 across twenty shuffles. The most important feature in the model, measured honestly, is one and a half standard deviations from mattering at all — an importance *ranking* over features none of which is demonstrably important, which is the usual situation on daily bars and almost never stated. The substitution experiment then shows why even the ordering cannot be trusted. The twin volatility estimators split their credit almost exactly in half, 0.072 and 0.071 — not because each contributes half the information but because the forest, offered identical information under two names, takes it from whichever column the split-sampling happens to serve up. Delete `f_vol_21` outright and the model's skill does not fall — the AUC *rises*, noise-level, from 0.492 to 0.497 — while `f_park`'s share climbs as it absorbs orphaned splits. The inheritance is partial rather than total for an instructive reason: volatility information also lives in `f_cs`, `f_range`, and `f_vix` (lesson one counted eight pairs above 0.8), so the deleted feature's credit disperses across the whole clique. The operational rule: MDI describes what the model *did in training*; permutation describes what it *needs at test time*; neither answers "what drives markets," and on correlated features both will happily rank members of a clique whose union matters and whose members are interchangeable. Feature *selection* built on these rankings inherits every one of these lies — [the appendix](../appendix/part-14-model-selection/04-feature-selection.md) treats the general problem.

## Calibration: a probability you can size with

The scoreboard graded thresholded decisions, but [lesson four](04-reinforcement-learning-and-meta-labeling.md) will want the probabilities themselves — a P(up) of 0.6 should mean *something happens 60% of the time*, or sizing on it is arithmetic on fiction. Boosted trees are notorious here: trained to rank, pushed toward confident scores by the loss, their raw outputs are not frequencies. The diagnosis is the reliability curve — bin predictions, compare each bin's claim to its realized frequency — and the repair is isotonic regression fitted on a held-out slice: 2017–2020 calibrates, 2021+ judges.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
X, y, w = spy[feat], (spy.y_tb > 0).astype(int), spy.w
tr = spy.index <= "2016"
cal = (spy.index >= "2017") & (spy.index <= "2020")    # calibration slice
te = spy.index >= "2021"                               # final evaluation

m = lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                       seed=42, deterministic=True, force_row_wise=True,
                       num_threads=1, verbosity=-1).fit(X[tr], y[tr], sample_weight=w[tr])
p_cal, p_te = m.predict_proba(X[cal])[:, 1], m.predict_proba(X[te])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y[cal])
q_te = iso.predict(p_te)
print(f"Brier on 2021+: raw {brier_score_loss(y[te], p_te):.4f}, "
      f"isotonic {brier_score_loss(y[te], q_te):.4f}, "
      f"always-base-rate {brier_score_loss(y[te], np.full(te.sum(), y[tr].mean())):.4f}")

bins = np.linspace(0.3, 0.8, 6)
mid = (bins[:-1] + bins[1:]) / 2
real_raw = [y[te][(p_te >= a) & (p_te < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
real_iso = [y[te][(q_te >= a) & (q_te < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
hi = (p_te >= 0.6)
print(f"raw scores span [{p_te.min():.2f}, {p_te.max():.2f}]; 2021+ base rate {y[te].mean():.3f}")
print(f"days with raw P(up) >= 0.60: {hi.sum()}, realized up-rate {y[te][hi].mean():.3f}")

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot([0.3, 0.8], [0.3, 0.8], ls="--", color="gray")
ax.plot(mid, real_raw, marker="o", label="raw")
ax.plot(mid, real_iso, marker="s", label="isotonic")
ax.set_xlabel("predicted P(up)")
ax.set_ylabel("realized frequency")
ax.legend()
plt.show()
# => Brier on 2021+: raw 0.4478, isotonic 0.2539, always-base-rate 0.2441
#    raw scores span [0.00, 0.94]; 2021+ base rate 0.587
#    days with raw P(up) >= 0.60: 69, realized up-rate 0.652
```

The Brier scores tell the story in three steps, and the middle step is the honest one. Raw LightGBM probabilities score 0.4478 — for scale, predicting 0.5 every day scores 0.25, so the raw model is *substantially worse than admitted ignorance*: its scores sweep the full [0.00, 0.94] range with a confidence its 0.48 AUC never earned, and Brier punishes confident wrongness quadratically. Isotonic recalibration collapses that bravado toward the base rate and cuts the score to 0.2539 — a repair of enormous size — and yet still lands a hair *above* 0.2441, which is what always predicting the training base rate scores. Read that ordering precisely, because it is the calibration lesson in one line: recalibration made the model's probabilities *honest*, and honesty revealed there was almost nothing behind them — a perfectly calibrated near-zero-skill model converges to quoting the base rate with tiny wiggles. The 69 high-confidence days show the wiggles are not quite empty (claimed ≥60%, realized 65.2% against a 58.7% base rate — directionally right, though 69 observations carry a six-point standard error), and that residue is the raw material [meta-labeling](04-reinforcement-learning-and-meta-labeling.md) will try to harvest. But the operational rule stands: never size on a raw ensemble score; calibrate on embargoed data first, then let the calibrated number tell you — as it does here — how little conviction you actually own.

!!! warning "The baseline is not a formality; it is the null hypothesis with a training budget"
    Every result in this lesson was legible only because something dumber stood next to it: the unlimited tree against always-up, the forest against its own average tree, the boosters against a referee that said stop, the grid against its median config, and all of it against a logistic regression and a buy-and-hold line. Run the ensemble without the baseline and you get numbers that feel like progress — 0.492! calibrated! tuned! — with nothing to reveal that a one-second linear model and a no-model both did better. The baseline is not there to be beaten politely; it is the null hypothesis, and the ensemble's job is to reject it with a margin that survives costs, noise, and the holdout. On this dataset it could not, and the honest report says so. That sentence — *the added complexity did not earn its keep* — is not a failed project; it is the single most common true finding in quantitative ML, and teams that cannot publish it internally end up trading its negation.

!!! abstract "Key takeaways"
    - An unrestricted tree hit 1.000 train / 0.490 test accuracy with 475 leaves; the depth-3 tree's respectable 0.606 still lost to the 2017+ base rate of 0.632 — accuracy is a rigged metric under drift, so AUC does the grading.
    - The forest's variance reduction worked (per-day tree spread 0.198 → 0.075) and its AUC of 0.492 matched the 0.497 mean of its own trees: averaging removes variance, not noise, and noise was the constraint.
    - Given an embargoed validation set, XGBoost early-stopped at round 0 and LightGBM at round 1 of 2000 offered; forced to 300 rounds they reached train AUC 0.946/0.992 and test AUC 0.477/0.467.
    - A purged, embargoed 5-fold grid search spread nine configs across 0.440–0.462 CV AUC; the CV champion scored 0.398 on the untouched 2022+ holdout, *below* the median config's 0.414 — hyperparameter selection is Part IV's overfitting story at one remove.
    - On the 2017+ scoreboard the logistic baseline (net Sharpe 0.38) beat the forest (0.17), LightGBM (0.20), and XGBoost (−0.01); buy-and-hold's 0.71 beat everything, tsmom (0.18) included.
    - The top permutation importance was 0.0077 ± 0.0054 AUC — the "most important" feature is ~1.4 sigma from irrelevant — and deleting `f_vol_21` moved AUC 0.492 → 0.497 while its 0.97-correlated twin absorbed its splits: importance on correlated features is bookkeeping, not markets.
    - Raw LightGBM scores earned a Brier of 0.4478 (worse than always saying 0.5); isotonic calibration on an embargoed slice repaired it to 0.2539, revealing a model whose honest form barely beats quoting the base rate (0.2441).

## Where this goes next

Tree ensembles were the strongest claim tabular ML had on this dataset, and the verdict came back: the machinery is superb, the information is not there, and the honest baseline held the line. The obvious escalation is capacity and structure — models that read *sequences* rather than rows, learn their own features, and power every headline the field generates. [Deep Learning](03-deep-learning.md) gives feed-forward networks, LSTMs, temporal convolutions, and a transformer the same purged splits, the same weights, and the same scoreboard, then runs the experiment that explains the outcome: how many samples does a deep model need before it separates signal from noise at this signal-to-noise ratio — and how many do daily bars actually offer?
