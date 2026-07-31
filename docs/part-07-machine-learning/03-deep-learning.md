# Deep Learning

[Tree Ensembles](02-tree-ensembles.md) ended with a scoreboard on which a one-second logistic regression out-earned five hundred trees and two gradient-boosting libraries, and an early-stopping referee that declined to boost at all. The natural objection is that trees were the wrong instrument: they see rows, not *sequences*; they combine hand-made features instead of learning their own; and every headline achievement in machine learning for a decade — vision, language, protein folding — came from the family this lesson finally deploys. Deep learning's claim on finance is specific: give a network the raw temporal structure and enough capacity, and it will discover features that lesson one's rolling windows never imagined.

The claim gets a fair trial and the same courtroom. Four architectures — a feed-forward network, an LSTM, a temporal convolutional network, and a transformer — train on the same frozen `data/part7.parquet`, the same uniqueness weights, the same 2016/2017 chronological split, and are graded by the same AUC and the same [cost model](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md). Every torch block pins its seeds (`manual_seed(0)`), runs single-threaded, and enables deterministic algorithms, so the numbers reproduce; every network is deliberately small — a few thousand parameters, thirty epochs, seconds on a CPU — because, as the middle of this lesson demonstrates with a controlled experiment, capacity was never going to be the binding constraint. Sample size is, and daily bars cannot pay.

## A feed-forward net meets the same four thousand rows

The multilayer perceptron is deep learning's null model: the nineteen features in, two hidden layers of ReLUs, one logit out. It can represent every interaction the forest found and smooth ones the forest cannot, and its parameter count — the first number printed below — is modest by any standard except the one that matters: parameters *per training row*.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
tr, te = spy.index <= "2016", spy.index >= "2017"
mu, sd = spy[feat][tr].mean(), spy[feat][tr].std()      # train-only scaling, lesson one's rule
X = torch.tensor(((spy[feat] - mu) / sd).values, dtype=torch.float32)
y = torch.tensor((spy.y_tb > 0).values, dtype=torch.float32)
w = torch.tensor(spy.w.values, dtype=torch.float32)

net = torch.nn.Sequential(torch.nn.Linear(19, 32), torch.nn.ReLU(),
                          torch.nn.Linear(32, 16), torch.nn.ReLU(),
                          torch.nn.Linear(16, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[tr], y[tr], w[tr]), batch_size=256,
    shuffle=True, generator=torch.Generator().manual_seed(0))
for epoch in range(30):
    for xb, yb, wb in dl:
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            net(xb).squeeze(-1), yb, reduction="none") * wb).mean()
        loss.backward()
        opt.step()

with torch.no_grad():
    p_tr, p_te = net(X[tr]).squeeze(-1), net(X[te]).squeeze(-1)
n_par = sum(p.numel() for p in net.parameters())
print(f"MLP 19-32-16-1: {n_par:,} parameters vs {tr.sum():,} training rows "
      f"({n_par / tr.sum():.2f} per row)")
print(f"train AUC {roc_auc_score(y[tr], p_tr):.3f}   test AUC {roc_auc_score(y[te], p_te):.3f}   "
      f"(lightgbm 0.481, logistic 0.478 on this split)")
# => MLP 19-32-16-1: 1,185 parameters vs 3,773 training rows (0.31 per row)
#    train AUC 0.720   test AUC 0.448   (lightgbm 0.481, logistic 0.478 on this split)
```

Note first what the scaling line does: the mean and standard deviation come from *training rows only* — the full-sample z-score was lesson one's planted leak, and here is the habit that avoids it. Then read the result as a ratio. A third of a parameter per row sounds safe; it is not, because [lesson one](01-feature-engineering-for-ml.md) showed the 3,773 rows carry the evidence of a few hundred independent observations, so the effective ratio is closer to three parameters per fact. The network spent them memorizing: train AUC 0.720 against test 0.448 is the same signature the 475-leaf tree and the forced boosters printed, achieved with a smoother pen. The gap to LightGBM's 0.481 and the logistic's 0.478 is inside the noise that [lesson two's importance section](02-tree-ensembles.md) measured — the honest reading is not "the MLP is worse" but "a fourth model family has now converged to the same coin flip." Everything the deeper architectures add from here is structure on top of this baseline, purchased with more parameters against the same few hundred facts.

## LSTMs buy memory with data you do not have

The recurrent pitch is that markets are sequences, and a model that carries state — the LSTM's gated memory cell, designed to remember and forget on cue — should read a 21-day window the way a trader reads a chart, not the way a spreadsheet reads a row. So the input changes shape: each observation becomes the last 21 days of all nineteen features, 399 numbers, and the network must compress that history into sixteen hidden units before predicting the same label as before.

```python
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
tr_m, te_m = spy.index <= "2016", spy.index >= "2017"
mu, sd = spy[feat][tr_m].mean(), spy[feat][tr_m].std()
Xf = ((spy[feat] - mu) / sd).values.astype("float32")
yf = (spy.y_tb > 0).values.astype("float32")
wf = spy.w.values.astype("float32")

L = 21
idx = np.arange(L - 1, len(Xf))                      # window [t-20 .. t] predicts label at t
Xs = torch.tensor(np.stack([Xf[i - L + 1:i + 1] for i in idx]))
ys, ws = torch.tensor(yf[idx]), torch.tensor(wf[idx])
tr, te = torch.tensor(tr_m[idx]), torch.tensor(te_m[idx])

class LSTMNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(19, 16, batch_first=True)
        self.head = torch.nn.Linear(16, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

net = LSTMNet()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xs[tr], ys[tr], ws[tr]), batch_size=256,
    shuffle=True, generator=torch.Generator().manual_seed(0))
t0 = time.perf_counter()
for epoch in range(30):
    for xb, yb, wb in dl:
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            net(xb).squeeze(-1), yb, reduction="none") * wb).mean()
        loss.backward()
        opt.step()
dt = time.perf_counter() - t0

with torch.no_grad():
    p_tr, p_te = net(Xs[tr]).squeeze(-1), net(Xs[te]).squeeze(-1)
n_par = sum(p.numel() for p in net.parameters())
print(f"LSTM(16) over 21-day windows: {n_par:,} parameters, {dt:.0f}s to train")
print(f"train AUC {roc_auc_score(ys[tr], p_tr):.3f}   test AUC {roc_auc_score(ys[te], p_te):.3f}")
# => LSTM(16) over 21-day windows: 2,385 parameters, 1s to train
#    train AUC 0.878   test AUC 0.451
```

The window inflated the input twenty-one-fold and the information not at all — that asymmetry is the whole result. Most of what a 21-day feature window contains is already summarized *inside* the features themselves: `f_ret_21` is the window's return, `f_vol_21` its volatility, `f_acorr` its serial structure. The LSTM is handed the ingredients of lesson one's kitchen and asked to rediscover the recipes, and with 2,385 parameters against the same few hundred effective facts, what it discovers instead is the training set: 0.878 in sample — the worst overfit yet — and 0.451 out. This is the recurring economics of sequence models on daily bars: recurrence is a *data amplifier in reverse*, multiplying the dimensionality of each observation while the number of observations stays fixed. Where LSTMs genuinely earn their reputation — language, sensor streams, order-flow sequences at tick resolution — the sequence dimension brings *new* observations by the million, not the same 3,773 rows viewed through a wider aperture. On the one honest metric, one second of training bought three points of AUC *below* the one-second logistic baseline.

## Temporal convolutions: causality as architecture

The temporal convolutional network treats the sequence not as state to be carried but as geometry to be filtered: stacks of one-dimensional convolutions whose dilations double at each layer, so four layers of kernel-3 filters see $1 + 2(1+2+4+8) = 31$ days. Two properties recommend it over the LSTM. It trains in parallel rather than step-by-step, and — the detail this course cares most about — its causality is *structural*: each layer pads only on the left, so no filter can touch a future value even by accident. Lesson one audited leakage; the TCN makes the audited property a compile-time fact.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
tr_m, te_m = spy.index <= "2016", spy.index >= "2017"
mu, sd = spy[feat][tr_m].mean(), spy[feat][tr_m].std()
Xf = ((spy[feat] - mu) / sd).values.astype("float32")
yf = (spy.y_tb > 0).values.astype("float32")
wf = spy.w.values.astype("float32")

L = 31                                              # receptive field 1 + 2*(1+2+4+8) = 31
idx = np.arange(L - 1, len(Xf))
Xs = torch.tensor(np.stack([Xf[i - L + 1:i + 1] for i in idx])).transpose(1, 2)
ys, ws = torch.tensor(yf[idx]), torch.tensor(wf[idx])
tr, te = torch.tensor(tr_m[idx]), torch.tensor(te_m[idx])

class TCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        chans, layers = [19, 16, 16, 16, 16], []
        for i, d in enumerate([1, 2, 4, 8]):
            layers += [torch.nn.ConstantPad1d((2 * d, 0), 0.0),   # left-pad only: causal
                       torch.nn.Conv1d(chans[i], chans[i + 1], 3, dilation=d),
                       torch.nn.ReLU()]
        self.stack = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(16, 1)
    def forward(self, x):
        return self.head(self.stack(x)[:, :, -1])

net = TCN()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xs[tr], ys[tr], ws[tr]), batch_size=256,
    shuffle=True, generator=torch.Generator().manual_seed(0))
for epoch in range(30):
    for xb, yb, wb in dl:
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            net(xb).squeeze(-1), yb, reduction="none") * wb).mean()
        loss.backward()
        opt.step()

with torch.no_grad():
    p_tr, p_te = net(Xs[tr]).squeeze(-1), net(Xs[te]).squeeze(-1)
n_par = sum(p.numel() for p in net.parameters())
print(f"TCN, 4 causal blocks, dilations 1/2/4/8: {n_par:,} parameters, "
      f"receptive field 31 days")
print(f"train AUC {roc_auc_score(ys[tr], p_tr):.3f}   test AUC {roc_auc_score(ys[te], p_te):.3f}")
# => TCN, 4 causal blocks, dilations 1/2/4/8: 3,297 parameters, receptive field 31 days
#    train AUC 0.841   test AUC 0.487
```

The receptive-field arithmetic in the comment is worth internalizing, because it is the leakage discipline of lesson one expressed as architecture: a value at position $t$ in the output can depend on positions $t-30$ through $t$ and *provably nothing later*, since every convolution's padding sits entirely in the past. Compare that guarantee to the audit lesson one had to run — recompute and diff — and you see the design principle: where possible, make look-ahead bias unrepresentable rather than merely detected, the same move [Part V](../part-05-backtesting-engine/01-architecture-and-event-driven-design.md) made when its event queue refused to deliver the future. The scoreboard entry, meanwhile, continues the pattern: 0.841 in training, 0.487 out — the best of the deep cohort so far by a nose, still below the linear baseline, still a coin. Architecture chose *which* noise to memorize (local shapes rather than the LSTM's long dependencies), and on data this thin, that choice is the only degree of freedom the network actually exercised.

## A transformer, with the hype removed

Attention replaced recurrence everywhere else, so it arrives here with the largest reputation and the largest parameter count. The mechanism is genuinely different: instead of compressing history through a bottleneck (LSTM) or filtering it locally (TCN), each day in the window *attends* to every other day, learning which past days matter for today's prediction. One encoder block, model width 32, four heads — a transformer at the smallest scale that still deserves the name. One honesty note before running it: our windows end at the prediction date, so full self-attention within the window touches only the past and no causal mask is needed — the mask matters when a single forward pass predicts at every position, [a configuration the execution-RL module uses](../advanced/06-rl-for-execution.md).

```python
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
tr_m, te_m = spy.index <= "2016", spy.index >= "2017"
mu, sd = spy[feat][tr_m].mean(), spy[feat][tr_m].std()
Xf = ((spy[feat] - mu) / sd).values.astype("float32")
yf = (spy.y_tb > 0).values.astype("float32")
wf = spy.w.values.astype("float32")

L = 21
idx = np.arange(L - 1, len(Xf))
Xs = torch.tensor(np.stack([Xf[i - L + 1:i + 1] for i in idx]))
ys, ws = torch.tensor(yf[idx]), torch.tensor(wf[idx])
tr, te = torch.tensor(tr_m[idx]), torch.tensor(te_m[idx])

class Former(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(19, 32)
        self.pos = torch.nn.Parameter(torch.randn(1, L, 32) * 0.02)
        self.enc = torch.nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.1)
        self.head = torch.nn.Linear(32, 1)
    def forward(self, x):
        z = self.enc(self.proj(x) + self.pos)       # window ends at t: all attention is past
        return self.head(z[:, -1])

net = Former()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xs[tr], ys[tr], ws[tr]), batch_size=256,
    shuffle=True, generator=torch.Generator().manual_seed(0))
for epoch in range(30):
    for xb, yb, wb in dl:
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            net(xb).squeeze(-1), yb, reduction="none") * wb).mean()
        loss.backward()
        opt.step()

net.eval()
with torch.no_grad():
    p_tr, p_te = net(Xs[tr]).squeeze(-1), net(Xs[te]).squeeze(-1)
n_par = sum(p.numel() for p in net.parameters())
print(f"1-block transformer encoder, d_model 32, 4 heads: {n_par:,} parameters")
print(f"train AUC {roc_auc_score(ys[tr], p_tr):.3f}   test AUC {roc_auc_score(ys[te], p_te):.3f}")
# => 1-block transformer encoder, d_model 32, 4 heads: 9,889 parameters
#    train AUC 0.872   test AUC 0.449
```

Ten thousand parameters — eight times the MLP — and the now-familiar signature: 0.872 memorized, 0.449 delivered, statistically the same coin every architecture in this lesson has flipped. The point of running it anyway is to close a rhetorical escape hatch. When tree ensembles failed in lesson two, "you should have used deep learning" remained sayable; when the MLP failed, "you ignored the sequence" remained; after the LSTM, "recurrence is obsolete, use attention." Each escalation has now been purchased and tested under identical, honest conditions, and each delivered the same verdict, which means the verdict was never about the architecture. Transformers dominate language because language offers billions of sequences whose structure is dense, compositional, and stationary enough to transfer. A daily bar series offers six thousand rows of mostly noise, and attention over noise is an expensive way to average it. The small `dropout=0.1` and the `net.eval()` call before scoring are the block's only other lessons — regularization did not save it, and a network scored in training mode leaks dropout randomness into its predictions, a bug that produces irreproducible tearsheets and is worth one sentence of paranoia forever.

## Why daily bars starve deep models

Four architectures, four coins. The diagnosis this lesson owes you is *quantitative*: how much data would these models need before they worked? The experiment: a synthetic market whose signal-to-noise ratio is tuned to daily-bar reality — nineteen features of which two interact and one adds a weak linear term, calibrated so that even an oracle who knows the true model scores only 0.569 AUC, about what a good daily-frequency desk claims. Because the generator is known, we can draw as many rows as we like and watch both model families climb toward the ceiling:

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def make_market(n, rng):
    """19 noise features; the signal is an interaction of two, scaled to daily-bar SNR."""
    X = rng.standard_normal((n, 19)).astype("float32")
    logit = 0.25 * X[:, 0] * X[:, 1] + 0.10 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("float32")
    return X, y, logit

rng = np.random.default_rng(0)
Xte, yte, logit_te = make_market(100_000, rng)
print(f"oracle AUC (knows the true model): {roc_auc_score(yte, logit_te):.3f}")

def mlp_auc(Xn, yn, epochs=10):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(19, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xn), torch.tensor(yn)),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(0))
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                net(xb).squeeze(-1), yb).backward()
            opt.step()
    with torch.no_grad():
        return roc_auc_score(yte, net(torch.tensor(Xte)).squeeze(-1))

def gbm_auc(Xn, yn):
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                           seed=42, deterministic=True, force_row_wise=True,
                           num_threads=1, verbosity=-1).fit(Xn, yn)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

ns = [1_000, 4_000, 16_000, 64_000, 250_000]
curves = {"mlp": [], "lightgbm": []}
for n in ns:
    Xn, yn, _ = make_market(n, np.random.default_rng(1))
    curves["mlp"].append(mlp_auc(Xn, yn))
    curves["lightgbm"].append(gbm_auc(Xn, yn))
    print(f"n = {n:>7,}: mlp {curves['mlp'][-1]:.3f}   lightgbm {curves['lightgbm'][-1]:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
for k, v in curves.items():
    ax.plot(ns, v, marker="o", label=k)
ax.axhline(roc_auc_score(yte, logit_te), ls="--", color="gray", label="oracle")
ax.axvline(3_773, ls=":", color="tab:red", label="SPY training rows")
ax.set_xscale("log")
ax.set_xlabel("training rows")
ax.set_ylabel("test AUC")
ax.legend()
plt.show()
# => oracle AUC (knows the true model): 0.569
#    n =   1,000: mlp 0.511   lightgbm 0.512
#    n =   4,000: mlp 0.523   lightgbm 0.514
#    n =  16,000: mlp 0.555   lightgbm 0.534
#    n =  64,000: mlp 0.563   lightgbm 0.550
#    n = 250,000: mlp 0.565   lightgbm 0.555
```

This is the most important printout in the lesson, because for once the deep model is *vindicated* — and the vindication is the indictment. Given 250,000 rows, the same MLP that flailed on SPY closes to within four thousandths of the oracle: 0.565 against a ceiling of 0.569, learning a multiplicative interaction that is exactly the kind of structure networks exist to find (and doing it better than LightGBM's 0.555, since trees approximate smooth interactions with staircases). The architecture works. Now find the red dotted line in the figure — SPY's 3,773 training rows — and read off what the same architecture achieves there: 0.523, roughly a *third* of the available skill, indistinguishable in practice from the noise floor. The signal in this synthetic market never changes, no regimes shift, no labels overlap, every row is independent — conditions vastly kinder than real markets — and the data budget of a daily-bar problem still starves the model of two-thirds of what is learnable. Real markets are crueler: the [effective sample](01-feature-engineering-for-ml.md) is a tenth of the row count and the signal drifts while you learn. The conclusion is arithmetic, not opinion: at this SNR, learnability begins in the tens of thousands of independent rows, and a quarter century of daily bars supplies four thousand.

## Where deep learning earns its keep

The learning curve names the cure as clearly as the disease: get more rows. There are three honest ways. Go *intraday* — a year of 5-minute bars is roughly 19,700 observations, so 25 years is nearly half a million, the top of the curve above ([Alternative Data](../advanced/07-alternative-data.md) covers the vendors, [GPU Acceleration](../advanced/08-gpu-acceleration-cuda.md) the compute this actually requires). Go *alternative* — text, filings, satellite imagery, where deep learning is not one option but the only serious extractor. Or go *cross-sectional* — trade many assets at once, so each date contributes many labels. The third is testable with data already on disk: Part IV's cache holds twelve sector and country ETFs, and the natural cross-sectional question — will this asset beat the median of its peers over the next week? — is [xsmom's](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) game, restated as classification.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

px = pd.read_parquet("data/part4.parquet").drop(columns=["VIX", "VIX3M"])
r = np.log(px).diff()

fwd5 = np.log(px).diff(5).shift(-5)
beat = fwd5.gt(fwd5.median(axis=1), axis=0)         # the cross-sectional game: beat the median
frames = []
for sym in px.columns:                              # 12 ETFs, features from closes alone
    s = r[sym].dropna()
    f = pd.DataFrame({"ret_5": s.rolling(5).sum(), "ret_21": s.rolling(21).sum(),
                      "ret_63": s.rolling(63).sum(), "ret_252": s.rolling(252).sum(),
                      "vol_21": s.rolling(21).std(),
                      "volratio": s.rolling(21).std() / s.rolling(63).std()})
    f["y"] = beat[sym].astype(float)
    f["sym"] = sym
    frames.append(f.dropna())
panel = pd.concat(frames).sort_index()
tr_p, te_p = panel[panel.index <= "2016"], panel[panel.index >= "2017"]
cols = ["ret_5", "ret_21", "ret_63", "ret_252", "vol_21", "volratio"]

def fit_mlp(dfr):
    mu, sd = dfr[cols].mean(), dfr[cols].std()
    Xn = torch.tensor(((dfr[cols] - mu) / sd).values, dtype=torch.float32)
    yn = torch.tensor(dfr.y.values, dtype=torch.float32)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xn, yn),
                                     batch_size=256, shuffle=True,
                                     generator=torch.Generator().manual_seed(0))
    for _ in range(20):
        for xb, yb in dl:
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                net(xb).squeeze(-1), yb).backward()
            opt.step()
    Xt = torch.tensor(((te_p[cols] - mu) / sd).values, dtype=torch.float32)
    with torch.no_grad():
        return roc_auc_score(te_p.y, net(Xt).squeeze(-1))

one = tr_p[tr_p.sym == "XLK"]
print(f"panel: {len(panel):,} rows across {px.shape[1]} ETFs; "
      f"train pool {len(tr_p):,} vs single-asset {len(one):,}")
print(f"same MLP, same 2017+ pooled test set:")
print(f"  trained on XLK alone   : AUC {fit_mlp(one):.3f}")
print(f"  trained on all 12 ETFs : AUC {fit_mlp(tr_p):.3f}")
# => panel: 72,569 rows across 12 ETFs; train pool 43,937 vs single-asset 3,773
#    same MLP, same 2017+ pooled test set:
#      trained on XLK alone   : AUC 0.497
#      trained on all 12 ETFs : AUC 0.505
```

Read the magnitude honestly before the direction. Pooling twelve assets multiplied the training set from 3,773 rows to 43,937 and moved AUC from 0.497 to 0.505 — the right sign, a real mechanism, and a small number. Small for two reasons the arithmetic predicts. Twelve highly correlated ETFs are nowhere near twelve independent bets per date — sector funds crash together, so breadth's [fundamental-law](../part-04-strategy-development/05-feature-and-signal-engineering.md) benefit is discounted by the average correlation — and six close-derived features give the network little that is genuinely cross-sectional to work with. The professional version of this trade takes the mechanism seriously at scale: thousands of stocks, not twelve funds; fundamental and text features, not six rolling windows; and at *that* scale the pooled panel reaches the millions of rows where section five's learning curve says deep models finally separate from the noise floor. That is not a hypothetical — cross-sectional equity is where neural networks genuinely operate in production — and it is the honest resolution of this lesson's tension: deep learning is not wrong for markets; it is wrong for *one asset's daily bars*, which is the setting this course's frozen caches can reproduce. The mechanism is real. The dose on display is homeopathic.

## The scoreboard, or benchmarking as discipline

One block now retrains everything this part has built — four shallow models from lesson two, four networks from this lesson — on the identical split and grades them on the identical strategy: long/flat SPY, next-day execution, 0.7 basis points per one-way trade. This is the moment a lazy lesson would declare a winner. Watch instead what the table actually licenses:

```python
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
X, y, w = spy[feat], (spy.y_tb > 0).astype(int), spy.w
tr_m, te_m = spy.index <= "2016", spy.index >= "2017"
c = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()["Close"]
r = np.log(c).diff()
COST = (0.5 + 0.2) * 1e-4                            # Part IV lesson seven, as before

mu, sd = X[tr_m].mean(), X[tr_m].std()
Xn = ((X - mu) / sd).values.astype("float32")
L = 21
idx = np.arange(L - 1, len(Xn))
Xseq = torch.tensor(np.stack([Xn[i - L + 1:i + 1] for i in idx]))

def train_net(net, Xt, sel, epochs=30):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt[sel], torch.tensor(y.values[sel], dtype=torch.float32),
                                       torch.tensor(w.values[sel], dtype=torch.float32)),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(0))
    for _ in range(epochs):
        for xb, yb, wb in dl:
            opt.zero_grad()
            ((torch.nn.functional.binary_cross_entropy_with_logits(
                net(xb).squeeze(-1), yb, reduction="none")) * wb).mean().backward()
            opt.step()
    net.eval()
    return net

class LSTMNet(torch.nn.Module):                      # the nets of sections one to four
    def __init__(self):
        super().__init__()
        self.lstm, self.head = torch.nn.LSTM(19, 16, batch_first=True), torch.nn.Linear(16, 1)
    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])

class TCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        chans, layers = [19, 16, 16, 16, 16], []
        for i, d in enumerate([1, 2, 4, 8]):
            layers += [torch.nn.ConstantPad1d((2 * d, 0), 0.0),
                       torch.nn.Conv1d(chans[i], chans[i + 1], 3, dilation=d), torch.nn.ReLU()]
        self.stack, self.head = torch.nn.Sequential(*layers), torch.nn.Linear(16, 1)
    def forward(self, x):
        return self.head(self.stack(x.transpose(1, 2))[:, :, -1])

class Former(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj, self.head = torch.nn.Linear(19, 32), torch.nn.Linear(32, 1)
        self.pos = torch.nn.Parameter(torch.randn(1, L, 32) * 0.02)
        self.enc = torch.nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                                    batch_first=True, dropout=0.1)
    def forward(self, x):
        return self.head(self.enc(self.proj(x) + self.pos)[:, -1])

rows = {}
shallow = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=50,
                                     random_state=42, n_jobs=4),
    "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             tree_method="hist", nthread=4, seed=42),
    "lightgbm": lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                   seed=42, deterministic=True, force_row_wise=True,
                                   num_threads=1, verbosity=-1),
}
for name, m in shallow.items():
    kw = {"logisticregression__sample_weight": w[tr_m]} if name == "logistic" else {"sample_weight": w[tr_m]}
    m.fit(X[tr_m], y[tr_m], **kw)
    rows[name] = pd.Series(m.predict_proba(X[te_m])[:, 1], index=spy.index[te_m])

torch.manual_seed(0)
mlp = train_net(torch.nn.Sequential(torch.nn.Linear(19, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 16), torch.nn.ReLU(),
                                    torch.nn.Linear(16, 1)),
                torch.tensor(Xn), np.nonzero(tr_m)[0])
with torch.no_grad():
    rows["mlp"] = pd.Series(torch.sigmoid(mlp(torch.tensor(Xn[te_m])).squeeze(-1)).numpy(),
                            index=spy.index[te_m])
for name, cls in {"lstm": LSTMNet, "tcn": TCN, "transformer": Former}.items():
    torch.manual_seed(0)
    net = train_net(cls(), Xseq, np.nonzero(tr_m[idx])[0])
    with torch.no_grad():
        p = torch.sigmoid(net(Xseq[torch.tensor(te_m[idx])]).squeeze(-1)).numpy()
    rows[name] = pd.Series(p, index=spy.index[idx][te_m[idx]])

print("model         AUC    net Sharpe   days long   (2017+, costs 0.7 bp/side)")
for name, p in rows.items():
    pos = (p > 0.5).astype(float)
    net_r = (pos.shift(1) * r.reindex(p.index) - pos.diff().abs() * COST).dropna()
    print(f"{name:12s} {roc_auc_score(y[spy.index.isin(p.index)], p):.3f}    "
          f"{np.sqrt(252) * net_r.mean() / net_r.std():+.2f}         {pos.mean():.0%}")
sig = np.sign(r.rolling(252).sum())
ts = (sig.shift(1) * r - sig.diff().abs() * COST).reindex(rows["mlp"].index).dropna()
bh = r.reindex(rows["mlp"].index).dropna()
print(f"{'tsmom':12s}        {np.sqrt(252) * ts.mean() / ts.std():+.2f}")
print(f"{'buy-hold':12s}        {np.sqrt(252) * bh.mean() / bh.std():+.2f}")
print(f"standard error of a Sharpe estimate on this window: ~{1 / np.sqrt(len(bh) / 252):.2f}")
# => model         AUC    net Sharpe   days long   (2017+, costs 0.7 bp/side)
#    logistic     0.478    +0.38         16%
#    forest       0.492    +0.17         38%
#    xgboost      0.471    -0.01         14%
#    lightgbm     0.481    +0.20         13%
#    mlp          0.448    +0.44         28%
#    lstm         0.476    +0.58         70%
#    tcn          0.496    +0.91         58%
#    transformer  0.487    +0.66         62%
#    tsmom               +0.18
#    buy-hold            +0.71
#    standard error of a Sharpe estimate on this window: ~0.34
```

A careless reading of this table would launch a fund: the TCN "beat the market," +0.91 against buy-and-hold's +0.71, and the deep cohort swept the top of the Sharpe column. Now apply the three disciplines that are this section's actual content. *First, read the exposure column against the Sharpe column.* The networks are long 58–70% of days; the shallow models 13–38%. On a window that was substantially one long rally, Sharpe differences of this size are mostly *how much of the rally you were in the room for* — the deep models' overconfident probabilities (recall lesson two's calibration verdict) cleared the 0.5 threshold far more often, and being wrong-but-long in a bull market pays. *Second, read the AUC column.* The TCN's forecasts rank days at 0.496 — a coin — and the MLP's at 0.448, the *worst on the board*, yet it out-Sharpes three models with better AUCs. When the forecast metric and the P&L metric disagree this badly, the P&L is telling you about the window, not the model. *Third, read the last line.* The standard error of a Sharpe ratio estimated from 8.5 years is roughly 0.34, so the entire scoreboard from −0.01 to +0.91 spans under three standard errors — approximately nothing distinguishes any two neighbors, and [Part IV's multiple-testing arithmetic](../part-04-strategy-development/08-validation-and-overfitting.md) reminds you that the maximum of eight noisy numbers is biased upward *because* it is a maximum. The benchmarking discipline, stated once: identical data, identical splits, identical costs, exposure reported next to performance, forecast metrics next to P&L metrics, and standard errors next to everything. Under that discipline the deep cohort's showing reads correctly — not "deep learning wins" but "eight coins, flipped over one bull market, in which the coins that said 'long' more often collected more drift."

!!! warning "Deep learning is a data-volume instrument, and daily bars are a data-poverty regime"
    The learning curve is the one exhibit in this lesson that generalizes, so carry it, not the scoreboard. The same network that flails at four thousand rows tracks the oracle at two hundred and fifty thousand — architecture was never the constraint, and no amount of architectural fashion (recurrence, convolution, attention, whatever arrives next) changes the row count. When someone shows you a deep model on daily bars, the first question is not "what architecture" but "how many independent observations" — and overlapping labels, regime drift, and correlated assets all shrink that number below the row count, usually by an order of magnitude. When the data volume is genuinely there — intraday, cross-sectional at thousands of names, alternative data — deep learning stops being hype and starts being the only tool that scales. Matching the instrument to the data regime is the entire skill.

!!! abstract "Key takeaways"
    - The MLP (1,185 parameters, 0.31 per row — but ~3 per *effective* row) posted train AUC 0.720, test 0.448: the 475-leaf tree's memorization signature with a smoother pen.
    - The LSTM inflated each observation 21× (train 0.878, test 0.451), the TCN made causality structural via left-padding (test 0.487, the deep cohort's best), and the 9,889-parameter transformer closed the escalation path at 0.449 — four architectures, one coin.
    - On a synthetic market with oracle AUC 0.569, the same MLP scores 0.523 at SPY's 3,773 rows and 0.565 at 250,000 — learnability at daily-bar SNR begins in the tens of thousands of independent rows, and a quarter century of daily bars supplies four thousand.
    - Pooling 12 ETFs (43,937 training rows) moved cross-sectional AUC from 0.497 to 0.505: breadth manufactures sample size, discounted by correlation — the mechanism that at thousands of names makes cross-sectional equity deep learning's real habitat.
    - The scoreboard inverted: the TCN's +0.91 net Sharpe (vs buy-and-hold's +0.71) came with a 0.496 AUC and 58% long exposure — bull-window beta harvested by overconfident probabilities, not forecasting skill.
    - A Sharpe estimated from 8.5 years carries a ~0.34 standard error; the whole eight-model board spans under three of them, and the maximum of eight noisy Sharpes is biased upward because it is a maximum.
    - Every torch block pinned seeds, ran single-threaded and deterministic, scaled by train-only statistics, and called `net.eval()` before scoring — the reproducibility scaffolding that makes these numbers checkable at all.

## Where this goes next

Supervised learning has now been tried at every capacity this dataset can carry, and the honest summary is that predicting *direction* from daily bars is a nearly empty problem: the best forecasts, shallow or deep, rank days indistinguishably from a coin. [Reinforcement Learning and Meta-Labeling](04-reinforcement-learning-and-meta-labeling.md) responds by changing the question twice. First it escalates — let an agent learn the whole trading *policy*, actions and all, and demonstrate with twenty seeded runs exactly why that fails at this signal-to-noise. Then it retreats to the move that actually works: stop asking models to find direction, hand them a primary signal that already has it — the tsmom rule this course has carried since Part IV — and ask only *when to believe it, and how much*. That division of labor is meta-labeling, and it is the first configuration in this part where the machine learning has a fighting chance of earning its keep.
