# Bayesian Optimization for Hyperparameters

[Part VII's grid search](../part-07-machine-learning/02-tree-ensembles.md) ended on a sentence with a debt in it: when a real search over a real space is warranted, Bayesian optimization does it with fewer evaluations — and nothing about a smarter searcher changes what is true. This module pays both halves. The first half is machinery worth owning: a Gaussian process that turns eight backtest runs into a posterior over the whole parameter surface, an acquisition function that prices the next evaluation before spending it, and the marginal likelihood that referees kernels without touching a holdout. The tooling is genuinely better than the grid — fewer evaluations, calibrated uncertainty, principled exploration. The second half is the reason this module lives beside [validation and overfitting](../part-04-strategy-development/08-validation-and-overfitting.md) rather than replacing it: a sample-efficient optimizer pointed at a noisy backtest metric is a sample-efficient noise maximizer, and every theorem that makes it find optima faster makes it find *false* optima faster too.

The data is the course's own: the [Part VII feature matrix](../part-07-machine-learning/01-feature-engineering-for-ml.md) in `data/part7.parquet`, the purged-fold objective from the tree-ensembles lesson, and the untouched 2022+ holdout that graded the nine-config grid. The uncomfortable result is pre-announced: a 60-trial Optuna search will beat the grid's cross-validated best, produce five different champions from five different seeds, and score **0.392** on the holdout the grid's champion scored 0.398 on — a smarter searcher, a larger search, and a worse ending, all for reasons an expected-maximum formula predicts to three decimal places before the holdout is ever opened.

!!! note "Versions"
    This module adds Optuna 4.9.0 to the Part VII stack (Python 3.12, NumPy 2.x, pandas 3.x, SciPy, scikit-learn 1.9, LightGBM 4.7). Everything runs from `data/part7.parquet` and `data/prices.parquet`; nothing here needs a GPU or a network connection.

## The objective is expensive, so every evaluation is chosen

Grid search treats evaluations as free. Nine LightGBM configurations under purged five-fold cross-validation cost about ten seconds on this course's data, and the grid spent them without thinking: three values of `num_leaves` crossed with three learning rates, most of them adjacent restatements of the same model. Scale the ambition — a five-parameter space, a slower model, an objective that is itself a full backtest — and the budget arithmetic turns hostile. A 10×10×10×10×10 grid of one-minute backtests is ten weeks of compute, of which the overwhelming majority is spent measuring regions the first hour already condemned.

Random search is the classical repair, and Bergstra and Bengio's result explains why it works: when only a few of the dimensions matter, random points project onto the important subspace without duplication, so sixty random configurations explore sixty distinct values of the one parameter that matters, where a 6×10 grid explored six. But random search shares the grid's deeper defect — evaluation $n$ learns nothing from evaluations $1$ through $n-1$. Bayesian optimization is the third step: keep a *model* of the objective, and let the model choose where to spend the next evaluation. The model is almost always a Gaussian process, the choosing rule is an acquisition function, and the pair converts a search history into a decision under uncertainty — which is why the machinery deserves respect *and* why it inherits every pathology of decision-making under noise that Parts III and IV catalogued.

## A Gaussian process is a prior over functions, not a curve fit

A Gaussian process says: before any data, the objective $f$ is a random function, and for any finite set of points $x_1,\dots,x_n$ the vector $(f(x_1),\dots,f(x_n))$ is jointly Gaussian with mean zero and covariance $K_{ij} = k(x_i, x_j)$. The kernel $k$ *is* the prior — it encodes how quickly the truth is allowed to wiggle. The workhorse choice for parameter surfaces is Matérn-5/2,

$$
k(x, x') \;=\; \sigma_f^2\left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right)\exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right),
\qquad r = \lVert x - x'\rVert,
$$

which produces functions rough enough to be honest about backtest surfaces — twice differentiable, no more — where the smoother squared-exponential kernel quietly assumes infinite differentiability and then over-trusts interpolation. This is [Part III's lesson about priors](../part-03-statistics/06-bayesian-methods-and-hmms.md) wearing new clothes: the kernel and its length-scale $\ell$ are *chosen*, they are not discovered, and every posterior below is conditional on that choice.

Conditioning is one identity. If observations $\mathbf y = f(X) + \varepsilon$ carry Gaussian noise $\varepsilon \sim \mathcal N(0, \sigma_n^2 I)$, then $(\mathbf y, f(x_*))$ is jointly Gaussian,

$$
\begin{pmatrix}\mathbf y \\ f(x_*)\end{pmatrix}
\;\sim\;
\mathcal N\!\left(\mathbf 0,\;
\begin{pmatrix} K + \sigma_n^2 I & \mathbf k_* \\ \mathbf k_*^\top & k(x_*,x_*)\end{pmatrix}\right),
$$

and one completion of the square in the joint density — condition a Gaussian on a Gaussian, the same algebra behind every Kalman update in the [filtering module](02-particle-and-kalman-filters.md) — yields the posterior in closed form:

$$
\mu_*(x_*) \;=\; \mathbf k_*^\top (K + \sigma_n^2 I)^{-1}\mathbf y,
\qquad
\sigma_*^2(x_*) \;=\; k(x_*, x_*) - \mathbf k_*^\top (K + \sigma_n^2 I)^{-1}\mathbf k_*.
$$

Read the second equation the way a risk manager would: the posterior variance does not depend on the observed *values* at all, only on the observed *locations*. The GP knows where it is ignorant purely from geometry. Here is the whole machine on a one-dimensional caricature of a strategy surface — true expected Sharpe as a function of a scaled lookback, eight noisy evaluations, and the posterior interrogated at three points:

```python
import numpy as np

rng = np.random.default_rng(0)

def truth(x):                                    # true expected Sharpe vs scaled lookback
    return 0.45 * np.exp(-0.5 * ((x - 0.55) / 0.18) ** 2) - 0.10

def matern52(xa, xb, ell=0.25, sf=0.35):
    r = np.abs(xa[:, None] - xb[None, :]) / ell
    return sf ** 2 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)

xs = rng.uniform(0, 1, 8)                        # eight backtests, already spent
noise = 0.05
ys = truth(xs) + noise * rng.standard_normal(8)

K = matern52(xs, xs) + noise ** 2 * np.eye(8)
L = np.linalg.cholesky(K)                        # never invert; solve
alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))

xstar = np.array([0.10, 0.44, 0.62])             # sampled region, unexplored gap, near incumbent
Ks = matern52(xs, xstar)
mu = Ks.T @ alpha
v = np.linalg.solve(L, Ks)
sd = np.sqrt(np.diag(matern52(xstar, xstar)) - np.sum(v ** 2, axis=0))

for x, m, s in zip(xstar, mu, sd):
    print(f"x = {x:.2f}: posterior {m:+.3f} +/- {s:.3f}   (truth {truth(np.array([x]))[0]:+.3f})")
# => x = 0.10: posterior -0.116 +/- 0.082   (truth -0.080)
#    x = 0.44: posterior +0.158 +/- 0.152   (truth +0.273)
#    x = 0.62: posterior +0.265 +/- 0.035   (truth +0.317)
```

The middle line is the one to sit with. At $x = 0.44$ — the widest unexplored gap, and as it happens the neighborhood of the true optimum — the posterior mean is badly wrong (+0.158 against a truth of +0.273), and the posterior *knows it*, attaching an error bar of ±0.152, four times wider than at the well-sampled $x = 0.62$. A curve fit would have handed you the wrong number with a straight face. The prior over functions hands you the wrong number with a confession attached, and the confession is what the acquisition function is about to spend.

## The marginal likelihood is the only honest kernel critic

Everything above was conditional on $\ell = 0.25$. Chosen differently, the same eight points tell different stories: a tiny length-scale lets the GP thread every observation and declare the gaps unknowable; a huge one flattens the surface into a line and calls the peak noise. The classical referee for this choice needs no validation set. Integrating the latent function out of the model gives the marginal likelihood of the data under the kernel itself,

$$
\log p(\mathbf y \mid X, \ell)
\;=\;
-\tfrac12\,\mathbf y^\top (K_\ell + \sigma_n^2 I)^{-1}\mathbf y
\;-\; \tfrac12 \log\lvert K_\ell + \sigma_n^2 I\rvert
\;-\; \tfrac{n}{2}\log 2\pi,
$$

and the two data-dependent terms are a fit–complexity ledger: the quadratic form rewards explaining the observations, the log-determinant charges for the volume of function space the kernel keeps in play. A kernel flexible enough to explain anything pays for it in determinant; a kernel too rigid to bend pays in fit. Maximizing this trade-off — type-II maximum likelihood — is Occam's razor executed by linear algebra:

```python
import numpy as np

rng = np.random.default_rng(0)

def truth(x):
    return 0.45 * np.exp(-0.5 * ((x - 0.55) / 0.18) ** 2) - 0.10

def matern52(xa, xb, ell, sf=0.35):
    r = np.abs(xa[:, None] - xb[None, :]) / ell
    return sf ** 2 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)

xs = rng.uniform(0, 1, 8)
noise = 0.05
ys = truth(xs) + noise * rng.standard_normal(8)

for ell in [0.02, 0.10, 0.25, 1.00, 4.00]:
    K = matern52(xs, xs, ell) + noise ** 2 * np.eye(8)
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, ys))
    lml = -0.5 * ys @ a - np.log(np.diag(L)).sum() - 4 * np.log(2 * np.pi)
    print(f"ell = {ell:4.2f}: log marginal likelihood {lml:7.2f}")
# => ell = 0.02: log marginal likelihood    0.47
#    ell = 0.10: log marginal likelihood    3.12
#    ell = 0.25: log marginal likelihood    5.44
#    ell = 1.00: log marginal likelihood   -1.66
#    ell = 4.00: log marginal likelihood  -17.29
```

The criterion peaks at $\ell = 0.25$ — the scale the surface was actually drawn at — and punishes the over-rigid kernels catastrophically (−17.29 at $\ell = 4$) while merely shrugging at the over-flexible one (0.47 at $\ell = 0.02$). That asymmetry is worth remembering: the marginal likelihood is better at detecting a kernel that *cannot* explain the data than one that explains it too easily, which is the statistical shape of most model-selection tools and the reason a maximized marginal likelihood is a hyperparameter search in its own right — one more layer of selection whose trial count belongs in the ledger the final sections audit.

## Expected improvement prices the value of ignorance

An acquisition function converts the posterior into a decision: given $\mu(x)$, $\sigma(x)$, and an incumbent best observation $f^+$, where is the next evaluation worth most? Expected improvement answers with an expectation you can integrate by hand. Improvement at $x$ is $\max(f(x) - f^+, 0)$; under the posterior $f(x) \sim \mathcal N(\mu, \sigma^2)$, split the integral at $f^+$, substitute $z = (f - \mu)/\sigma$, and the truncated-Gaussian moments collapse to

$$
\mathrm{EI}(x)
\;=\;
\bigl(\mu(x) - f^+\bigr)\,\Phi(z) \;+\; \sigma(x)\,\varphi(z),
\qquad
z \;=\; \frac{\mu(x) - f^+}{\sigma(x)},
$$

two terms with a trader's reading: the first is the exploitation leg — how much the mean already clears the incumbent, weighted by the probability it really does — and the second is the exploration leg, an option premium paid purely for variance. A point can win on either leg. The upper confidence bound $\mu(x) + \kappa\sigma(x)$ prices the same trade-off with one dial, and its theory (the GP-UCB schedule for $\kappa_t$ of Srinivas et al.) earns sublinear regret; EI's virtue is having no dial at all. Verify the formula against brute force, then watch the two legs argue:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

def truth(x):
    return 0.45 * np.exp(-0.5 * ((x - 0.55) / 0.18) ** 2) - 0.10

def matern52(xa, xb, ell=0.25, sf=0.35):
    r = np.abs(xa[:, None] - xb[None, :]) / ell
    return sf ** 2 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)

xs = rng.uniform(0, 1, 8)
noise = 0.05
ys = truth(xs) + noise * rng.standard_normal(8)
K = matern52(xs, xs) + noise ** 2 * np.eye(8)
L = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))

xstar = np.array([0.10, 0.44, 0.62])
Ks = matern52(xs, xstar)
mu = Ks.T @ alpha
v = np.linalg.solve(L, Ks)
sd = np.sqrt(np.diag(matern52(xstar, xstar)) - np.sum(v ** 2, axis=0))

f_best = ys.max()
z = (mu - f_best) / sd
ei = (mu - f_best) * stats.norm.cdf(z) + sd * stats.norm.pdf(z)

draws = mu[:, None] + sd[:, None] * np.random.default_rng(1).standard_normal((3, 4_000_000))
ei_mc = np.maximum(draws - f_best, 0).mean(axis=1)
for x, a, b in zip(xstar, ei, ei_mc):
    print(f"x = {x:.2f}: EI closed form {a:.4f}, Monte Carlo {b:.4f}")
print(f"incumbent f_best = {f_best:+.3f}, max |closed - MC| = {np.abs(ei - ei_mc).max():.6f}")
# => x = 0.10: EI closed form 0.0000, Monte Carlo 0.0000
#    x = 0.44: EI closed form 0.0212, Monte Carlo 0.0212
#    x = 0.62: EI closed form 0.0133, Monte Carlo 0.0133
#    incumbent f_best = +0.266, max |closed - MC| = 0.000008
```

Four million Monte Carlo draws agree with the two-term formula to the sixth decimal, which settles the algebra; the middle rows settle the philosophy. The unexplored gap at $x = 0.44$ carries a *lower* posterior mean than the near-incumbent point at $x = 0.62$ — +0.158 against +0.265 — and EI prices it **60% higher** anyway (0.0212 vs 0.0133), because the gap's ±0.152 of ignorance is worth more than the incumbent neighborhood's ±0.035 of confirmation. The acquisition function is paying for information, not for flattery, and on this seed the information is real: the gap is where the true optimum lives.

## A backtest metric is a noisy oracle, and the best print lies

Everything so far assumed the oracle whispers the truth plus small noise. A backtest metric is noisier than that — a Sharpe estimated on ten years carries a standard error near $1/\sqrt{10} \approx 0.3$, and a cross-validated AUC on overlapping labels is little better — and noise breaks the innocent-looking $f^+$ in the EI formula. The best *observation* is not the best *point*; it is the point whose noise draw was luckiest, which is why the honest incumbent under noise is the posterior mean at the best observed location (plug-in EI), why production systems re-evaluate incumbents instead of trusting them, and why the number a search *reports* and the number it *delivers* are different quantities. Watch all three diverge on a two-dimensional surface whose true optimum is a Sharpe of 0.350 and whose single-evaluation noise is 0.15 — twenty independent searches each way:

```python
import numpy as np
from scipy import stats

def truth(p):                                    # true Sharpe over a 2-D parameter box
    return (0.45 * np.exp(-0.5 * (((p[:, 0] - 0.55) / 0.18) ** 2
                                  + ((p[:, 1] - 0.40) / 0.22) ** 2)) - 0.10)

NOISE = 0.15                                     # one backtest's sampling noise
G = np.stack(np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 41)), -1).reshape(-1, 2)

def k2(a, b, ell=0.25, sf=0.35):
    r = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)) / ell
    return sf ** 2 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)

def gp_post(X, y, Xstar):
    L = np.linalg.cholesky(k2(X, X) + NOISE ** 2 * np.eye(len(X)))
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = k2(X, Xstar)
    v = np.linalg.solve(L, Ks)
    var = np.clip(k2(Xstar, Xstar).diagonal() - (v ** 2).sum(0), 1e-12, None)
    return Ks.T @ a, np.sqrt(var)

def run_bo(seed, n_evals=30, n_init=6):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n_init, 2))
    y = truth(X) + NOISE * r.standard_normal(n_init)
    for _ in range(n_evals - n_init):
        mu, sd = gp_post(X, y, G)
        z = (mu - y.max()) / sd
        ei = (mu - y.max()) * stats.norm.cdf(z) + sd * stats.norm.pdf(z)
        xn = G[np.argmax(ei)][None, :]
        X = np.vstack([X, xn])
        y = np.append(y, truth(xn) + NOISE * r.standard_normal(1))
    mu, _ = gp_post(X, y, G)
    return y.max(), truth(X[np.argmax(y)][None, :])[0], truth(G[np.argmax(mu)][None, :])[0]

def run_random(seed, n_evals=30):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n_evals, 2))
    y = truth(X) + NOISE * r.standard_normal(n_evals)
    return y.max(), truth(X[np.argmax(y)][None, :])[0]

seeds = np.random.SeedSequence(0).spawn(20)
bo = np.array([run_bo(s) for s in seeds])
rd = np.array([run_random(s) for s in seeds])

print(f"true optimum 0.350; single-evaluation noise sd {NOISE:.2f}; 30 evaluations each")
print(f"random search: reported best {np.median(rd[:, 0]):+.2f}, "
      f"true value of that point {np.median(rd[:, 1]):+.2f}")
print(f"BO, best observation: reported {np.median(bo[:, 0]):+.2f}, "
      f"true value {np.median(bo[:, 1]):+.2f}")
print(f"BO, posterior-mean argmax: true value {np.median(bo[:, 2]):+.2f}")
# => true optimum 0.350; single-evaluation noise sd 0.15; 30 evaluations each
#    random search: reported best +0.42, true value of that point +0.24
#    BO, best observation: reported +0.46, true value +0.27
#    BO, posterior-mean argmax: true value +0.31
```

Three verdicts in four lines. Random search *reports* +0.42 and *delivers* +0.24 — 0.18 of pure winner's curse, the max-of-thirty noise draws dressed as skill, [Part IV's expected-maximum arithmetic](../part-04-strategy-development/08-validation-and-overfitting.md) in miniature. Bayesian optimization's best observation is just as dishonest a reporter (+0.46 claimed, +0.27 true) — concentrating evaluations near the peak means concentrating noise draws there too. The one honest deliverable on the board is the third: the *posterior-mean* argmax truly earns +0.31, closest to the 0.350 ceiling, because averaging across repeated nearby evaluations is exactly what a posterior mean is for. The operational rule falls out directly, and it is the rule every later section enforces: **the surrogate's mean is the product; the best print is marketing.**

## TPE is not a Gaussian process, and mostly that does not matter

Optuna — the library this course reaches for — does not model $p(y \mid x)$ with a GP at all. Its tree-structured Parzen estimator models the *inverse*: split the trials at a quantile $\gamma$ of the objective, fit one density to the good configurations and another to the rest, and rank candidates by the ratio

$$
\alpha(x) \;\propto\; \frac{\ell(x)}{g(x)},
\qquad
\ell(x) = p(x \mid y \ge y_\gamma),\quad g(x) = p(x \mid y < y_\gamma),
$$

which Bergstra et al. showed is monotone in expected improvement under the split — the same option, priced from the other side of the ledger. The trade is practical, not philosophical: TPE natively handles integer, categorical, and conditional parameters (where a GP needs bespoke kernels), parallelizes without contortions, and scales past the few-hundred-trial mark where Cholesky factorizations start to bite; a GP squeezes more out of each evaluation in low-dimensional continuous boxes and is the substrate for the exotic acquisitions in BoTorch/Ax, with scikit-optimize the minimal GP option between them. On the surface the hand-rolled loop just searched, the imported machine tells the same story in five lines — including the same lie in its headline number:

```python
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def truth(x, y):
    return 0.45 * np.exp(-0.5 * (((x - 0.55) / 0.18) ** 2 + ((y - 0.40) / 0.22) ** 2)) - 0.10

NOISE = 0.15
rng = np.random.default_rng(0)

def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    return truth(x, y) + NOISE * rng.standard_normal()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
bx, by = study.best_params["x"], study.best_params["y"]
print(f"TPE 30 trials: reported best {study.best_value:+.2f} at ({bx:.2f}, {by:.2f}), "
      f"true value there {truth(bx, by):+.2f}")
# => TPE 30 trials: reported best +0.41 at (0.62, 0.22), true value there +0.20
```

`study.best_value` says +0.41; the truth at `study.best_params` is +0.20. Nothing malfunctioned — the sampler did its job, the noise did its job, and the reporting convention did its damage. Optuna will happily hand you the trials dataframe from which an honest estimate can be built; it will not build one for you. Carry that habit into the only experiment in this module that touches real money-shaped data.

## Sixty smart trials overfit faster than nine dumb ones

The stage is exactly as [the tree-ensembles lesson](../part-07-machine-learning/02-tree-ensembles.md) left it: SPY rows of `data/part7.parquet`, development years through 2021, the purged five-fold objective with a 21-day embargo, and a 2022+ holdout that has been opened once in this course's history — to grade a nine-config grid that scored CV 0.440–0.462, whose champion then delivered 0.398 against the median config's 0.414. The promise made there was that a smarter searcher would not change what is true. Sixty TPE trials over the same box, six times the grid's budget spent adaptively:

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
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
        aucs.append(roc_auc_score((dev.y_tb.iloc[te] > 0),
                                  m.predict_proba(dev[feat].iloc[te])[:, 1]))
    return np.mean(aucs)

def objective(trial):
    return cv_auc(trial.suggest_int("num_leaves", 7, 31),
                  trial.suggest_float("learning_rate", 0.02, 0.10, log=True))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60)           # about a minute of LightGBM
bp = study.best_params
vals = np.array([t.value for t in study.trials])
print(f"TPE, 60 trials: best CV AUC {study.best_value:.3f} "
      f"at ({bp['num_leaves']}, {bp['learning_rate']:.3f})")
print(f"trial values: min {vals.min():.3f}, median {np.median(vals):.3f}, max {vals.max():.3f}")

m = lgb.LGBMClassifier(n_estimators=300, num_leaves=bp["num_leaves"],
                       learning_rate=bp["learning_rate"], seed=42, deterministic=True,
                       force_row_wise=True, num_threads=1, verbosity=-1)
m.fit(dev[feat], (dev.y_tb > 0), sample_weight=dev.w)
print(f"untouched 2022+ holdout: TPE champion "
      f"{roc_auc_score((hold.y_tb > 0), m.predict_proba(hold[feat])[:, 1]):.3f}")
# => TPE, 60 trials: best CV AUC 0.468 at (31, 0.089)
#    trial values: min 0.447, median 0.458, max 0.468
#    untouched 2022+ holdout: TPE champion 0.392
```

Hold both lines at once. The search *worked*: 0.468 beats the grid's 0.462, found adaptively, with the trial distribution visibly concentrated in the profitable-looking corner (median 0.458 against the grid's 0.455). And the holdout *graded* it: **0.392** — below the grid champion's 0.398, below the grid median's 0.414, below every number the dumber procedure produced. Six times the search effort bought a better tournament and a worse model, exactly the trade [Part IV's PBO logic](../part-04-strategy-development/08-validation-and-overfitting.md) prices: when true differences between configurations are smaller than evaluation noise, a better optimizer climbs the noise more efficiently, and the reward for winning a rigged tournament is losing hardest afterward.

If the champion were a discovery rather than a draw, an independent search should rediscover it. Five searches, identical except for the sampler seed — about five minutes of compute, and the cheapest audit in this module:

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
dev = spy[spy.index <= "2021"]

def purged_folds(t1, n_splits=5, embargo=21):
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
        aucs.append(roc_auc_score((dev.y_tb.iloc[te] > 0),
                                  m.predict_proba(dev[feat].iloc[te])[:, 1]))
    return np.mean(aucs)

def objective(trial):
    return cv_auc(trial.suggest_int("num_leaves", 7, 31),
                  trial.suggest_float("learning_rate", 0.02, 0.10, log=True))

champs = []
for s in range(5):
    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=s))
    st.optimize(objective, n_trials=60)
    champs.append((st.best_params["num_leaves"],
                   round(st.best_params["learning_rate"], 3), round(st.best_value, 3)))
    print(f"seed {s}: champion {champs[-1]}")
print(f"distinct champions across 5 searches: {len(set(c[:2] for c in champs))}")
# => seed 0: champion (31, 0.089, 0.468)
#    seed 1: champion (22, 0.094, 0.474)
#    seed 2: champion (31, 0.093, 0.475)
#    seed 3: champion (26, 0.1, 0.476)
#    seed 4: champion (13, 0.075, 0.471)
#    distinct champions across 5 searches: 5
```

Five searches, **five different champions** — `num_leaves` anywhere from 13 to 31, agreement on nothing except that all five "optima" print between 0.468 and 0.476. A real optimum is a property of the objective; these are properties of the seed. The expected-maximum formula from [the deflated-Sharpe machinery](../part-04-strategy-development/08-validation-and-overfitting.md) makes the indictment quantitative — take the grid's nine configurations as nine draws from one plateau, ask what the best of $N$ such draws should print,

$$
\mathbb{E}[\max_N] \;\approx\; \mu_{\text{plateau}} + \sigma\Bigl[(1-\gamma_E)\,\Phi^{-1}\!\bigl(1-\tfrac1N\bigr) + \gamma_E\,\Phi^{-1}\!\bigl(1-\tfrac{1}{Ne}\bigr)\Bigr],
$$

with $\gamma_E$ the Euler–Mascheroni constant, and compare to what the tournaments actually printed:

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
dev = spy[spy.index <= "2021"]

def purged_folds(t1, n_splits=5, embargo=21):
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
        aucs.append(roc_auc_score((dev.y_tb.iloc[te] > 0),
                                  m.predict_proba(dev[feat].iloc[te])[:, 1]))
    return np.mean(aucs)

grid = np.array([cv_auc(lv, lr) for lv in [7, 15, 31] for lr in [0.02, 0.05, 0.10]])
med, sig = np.median(grid), grid.std(ddof=1)
print(f"nine grid configs: median {med:.3f}, sd {sig:.4f}, best {grid.max():.3f}")

def emax(n):                                     # E[max of n] of standard normals
    return ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n)
            + np.euler_gamma * stats.norm.ppf(1 - 1 / (n * np.e)))

for n, realized, label in [(9, grid.max(), "grid best"), (60, 0.468, "TPE best")]:
    print(f"E[best of {n:2d} equal players] = {med:.3f} + {sig:.4f} * {emax(n):.2f} "
          f"= {med + sig * emax(n):.3f}   (realized {label} {realized:.3f})")
# => nine grid configs: median 0.455, sd 0.0073, best 0.462
#    E[best of  9 equal players] = 0.455 + 0.0073 * 1.52 = 0.466   (realized grid best 0.462)
#    E[best of 60 equal players] = 0.455 + 0.0073 * 2.35 = 0.472   (realized TPE best 0.468)
```

This is the module's verdict, delivered without opening any holdout. Under the null that every configuration is the *same player* — one plateau, differences pure noise — the best of nine draws should print about 0.466, and the grid's champion printed 0.462. The best of sixty should print about 0.472, and TPE's champion printed 0.468. Both tournaments produced *slightly less* than what tournaments among identical players produce (the shortfall is itself informative: TPE's sixty trials are correlated, so they buy fewer than sixty independent draws). Charge the search for its history — sixty trials, or the five-times-sixty of the seed audit for anyone who ran it — and the deflation arithmetic returns the same answer the 2022 holdout returned for free: there is nothing here, and the arithmetic knew first.

## Diagnosing a search that converged on nothing

Optuna will report `study.best_value` with the same typography whether it found a real optimum or a lucky draw, so the diagnosis is yours to run, and three checks cover most of the failure surface. First, **re-evaluate the incumbent under perturbed validation** — same configuration, same data, ten different fold geometries — and compare the spread to the margin of victory:

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
dev = spy[spy.index <= "2021"]

def purged_folds(t1, n_splits, embargo):
    idx, n = t1.index, len(t1)
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        t_max = t1.iloc[a:b].max()
        train = [i for i in range(n) if (i < a and t1.iloc[i] < idx[a])
                 or (i >= b + embargo and idx[i] > t_max)]
        yield np.array(train), np.arange(a, b)

def cv_auc(leaves, lr, n_splits=5, embargo=21):
    aucs = []
    for tr, te in purged_folds(dev.t1, n_splits, embargo):
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=leaves, learning_rate=lr,
                               seed=42, deterministic=True, force_row_wise=True,
                               num_threads=1, verbosity=-1)
        m.fit(dev[feat].iloc[tr], (dev.y_tb.iloc[tr] > 0), sample_weight=dev.w.iloc[tr])
        aucs.append(roc_auc_score((dev.y_tb.iloc[te] > 0),
                                  m.predict_proba(dev[feat].iloc[te])[:, 1]))
    return np.mean(aucs)

geoms = [(4, 10), (4, 21), (5, 10), (5, 21), (5, 42),
         (5, 63), (6, 10), (6, 21), (7, 21), (8, 21)]
revals = np.array([cv_auc(31, 0.089, ns, em) for ns, em in geoms])  # the TPE champion
print(f"champion re-evaluated under 10 fold geometries: mean {revals.mean():.3f}, "
      f"sd {revals.std(ddof=1):.4f}, range [{revals.min():.3f}, {revals.max():.3f}]")
# => champion re-evaluated under 10 fold geometries: mean 0.470, sd 0.0099, range [0.460, 0.487]
```

The champion's own value swings from 0.460 to 0.487 — a range **larger than the entire 0.440-to-0.462 spread** that separated the grid's best configuration from its worst — when nothing changes but the referee's fold boundaries. Any ranking decided inside that band is decided by geometry, not by hyperparameters. Second, **read the incumbent trace**: the sixty-trial study improved its incumbent only four times, twice after trial ten — a search that plateaus immediately and then shops for noise is describing a flat objective, and the honest response to a flat objective is to *stop*, not to extend the budget. Third, **watch the box walls**: four of the five seed-audit champions put the learning rate within a whisker of the 0.10 boundary. An optimizer piling into a corner of the search box is usually reporting that the box, not the objective, binds — in a real tuning campaign that means widening the range and re-charging the ledger for the second search, and in this one it means the "optimum" is wherever variance was highest. Surrogate misfit runs in the same spirit (a GP whose marginal likelihood prefers absurd length-scales is telling you the surface is not GP-shaped — Section 3's tool turned inward), and all four checks share one property worth more than any of them: they consume no holdout. The holdout stays shut until the search history is *finished*, counted, and deflated — at which point, on this data, there is nothing left to grade.

!!! warning "A sample-efficient optimizer is a sample-efficient overfitter"
    Bayesian optimization changes how fast you climb; it cannot change what you are climbing. On an objective whose configuration differences are smaller than its evaluation noise — the normal case for backtest metrics — sixty adaptive trials bought a higher in-sample number, five irreproducible champions, and a worse holdout than nine dumb ones, and the expected-max formula priced the whole outcome in advance. Deflate by the trials you ran, including the searches you threw away, or the optimizer becomes the most efficient noise-harvesting machine you own.

!!! abstract "Key takeaways"
    - A GP posterior is geometry-aware honesty: at the widest unexplored gap it reported +0.158 ± 0.152 where truth was +0.273 — wrong mean, correct confession, and the confession is what acquisition functions spend.
    - The marginal likelihood referees kernels without a holdout: log ML peaked at 5.44 for the true length-scale 0.25, punished rigidity at −17.29, and barely penalized over-flexibility at 0.47 — it detects kernels that cannot fit far better than kernels that fit too easily.
    - The closed-form EI $(\mu-f^+)\Phi(z) + \sigma\varphi(z)$ matched four million Monte Carlo draws to 8 × 10⁻⁶, and priced a lower-mean unexplored point 60% above a higher-mean explored one (0.0212 vs 0.0133) — variance is an asset when information is the product.
    - Under 0.15 evaluation noise, random search reported +0.42 and truly delivered +0.24; BO's best observation reported +0.46 and delivered +0.27; BO's posterior-mean argmax delivered +0.31 of a 0.35 ceiling. The surrogate's mean is the product; the best print is marketing.
    - On the real Part VII objective, 60 TPE trials beat the grid in-sample (CV 0.468 vs 0.462) and lost to it out of sample (holdout 0.392 vs 0.398 champion, 0.414 median) — six times the budget, adaptively spent, made the overfit worse.
    - Five seeded re-runs of the same search produced five different champions (0.468–0.476), and the champion's own value ranged 0.460–0.487 across ten fold geometries — wider than the grid's entire best-to-worst spread, so the ranking was referee noise.
    - The expected-max-of-N formula predicted the tournaments to three decimals with no holdout: E[best of 9 equal players] 0.466 vs realized 0.462; E[best of 60] 0.472 vs realized 0.468. Both searches printed *at or below* the all-noise null.
    - Diagnostics that cost no holdout: re-evaluate the incumbent under perturbed folds, read the incumbent trace (4 improvements in 60 trials = a flat objective), and watch for champions piling against the search-box walls (4 of 5 did).

## Where this goes next

The accounting that ended this module — charge every search for every trial it ran, at the scale it ran them — becomes a systems problem the moment the trials stop fitting on one machine: [Distributed Backtesting](09-distributed-backtesting.md) runs the same expected-maximum arithmetic at ten-thousand-trial scale on a Ray cluster, where the formula stops being a caution and becomes the design constraint. The validation gauntlet this module leaned on at every turn is [Part IV, lesson eight](../part-04-strategy-development/08-validation-and-overfitting.md); the grid whose sentence this module served is in [Part VII's tree-ensembles lesson](../part-07-machine-learning/02-tree-ensembles.md); and when a tuned model finally does earn deployment, [Production ML](../part-07-machine-learning/05-production-ml.md) is where its search history gets written into the model registry next to its data hash — provenance for the day someone asks why the live numbers look nothing like `study.best_value`.
