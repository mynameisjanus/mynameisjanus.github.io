# Risk Measurement

[Production ML](../part-07-machine-learning/05-production-ml.md) closed the part that built a model and shipped it, and it closed by naming what the whole apparatus still could not do: manage more than one signal at a time. Six lessons of Part VI hardened the processes around a single strategy; five lessons of Part VII conditioned, monitored and versioned a single model. A desk is neither. A desk is many return streams drawing on one pool of capital, and the first question anyone asks about that pool is not *what will it earn* but *what is it currently risking* — a question with a specific numerical answer that must be produced before the market opens, defended in front of people who did not build the models, and revised when it turns out to have been wrong.

This lesson produces that answer. It assembles the course's strategies into a single panel, measures the volatility of what they add up to, prices the loss in the tail three different ways, and then does the thing most treatments skip: it *backtests the risk model itself*, discovers that the standard one fails at odds that underflow double precision, and works out which part of it is broken. The recurring finding is that risk numbers are estimates with their own error bars and their own failure modes, and that a book's risk is almost never distributed the way its position sizes suggest.

## The freeze: one panel for the whole part

Everything from here to the end of Part VIII runs off one file. The five strategies of [Part IV](../part-04-strategy-development/07-portfolio-construction-and-transaction-costs.md) plus the meta-labeled trend book of [Part VII](../part-07-machine-learning/04-reinforcement-learning-and-meta-labeling.md) become six daily net-of-cost return streams; the assets they trade become eighteen daily return columns for the covariance work of lessons three and four; VIX and its three-month cousin come along for regime context. Three construction details are not cosmetic. `shortvol` was defined monthly in variance points, so it is rebuilt here as a one-month variance swap struck at the prior month-end VIX and *marked to market daily* — an accounting change that must telescope back to Part IV's monthly number exactly, or the sleeve has been quietly redefined. `pairs` is charged for **both** legs of its round trip, since buying SPY and selling IVV is two trades, not one. And `xsmom` holds its monthly weights across daily returns. The block is the longest in the course, deliberately, because it is the entire part in one place:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
HS = {"SPY": 0.5, "IVV": 1.0, "TLT": 1.0, "GLD": 1.0, "SEC": 2.0}     # Part IV half-spreads, bp
COMM = 0.2

px = pd.read_parquet("data/prices.parquet")
p4 = pd.read_parquet("data/part4.parquet")
bars = pd.read_parquet("data/part5.parquet")
cal = px.index

# --- tsmom and tsmom_meta: Part VII lesson four's raw and gated books, unchanged ---
mat = pd.read_parquet("data/part7.parquet")
act = mat[mat.sig_tsmom != 0].sort_index()
feat = [k for k in act.columns if k.startswith("f_")]
y_meta = (np.sign(act.ret_5d) == act.sig_tsmom).astype(int)

def purged_folds(t1, n_splits=5, embargo=21):        # the folds of Part VII lesson two
    idx, n = t1.index, len(t1)
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        t_max = t1.iloc[a:b].max()
        train = [i for i in range(n) if (i < a and t1.iloc[i] < idx[a])
                 or (i >= b + embargo and idx[i] > t_max)]
        yield np.array(train), np.arange(a, b)

p_oof = pd.Series(np.nan, index=act.index)
for tr, te in purged_folds(act.t1):
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                           seed=42, deterministic=True, force_row_wise=True,
                           num_threads=1, verbosity=-1)
    m.fit(act[feat].iloc[tr], y_meta.iloc[tr], sample_weight=act.w.iloc[tr])
    p_oof.iloc[te] = m.predict_proba(act[feat].iloc[te])[:, 1]
act = act.assign(p_meta=p_oof.values)

raw_legs, meta_legs = {}, {}
for a in ["SPY", "TLT", "GLD"]:
    rs_a = np.log(bars.xs(a, axis=1, level=1).dropna()["Close"]).diff()
    sub = act[act.symbol == a]
    b = sub.sig_tsmom.reindex(rs_a.index).ffill()      # NaN before the sleeve's first signal
    p = sub.p_meta.reindex(rs_a.index).ffill()
    c = (HS[a] + COMM) * 1e-4
    for store, pos in [(raw_legs, b), (meta_legs, b * (p > 0.5))]:
        store[a] = (pos.shift(1) * rs_a - pos.diff().abs().fillna(0.0) * c).reindex(cal)
s_tsmom = pd.DataFrame(raw_legs).mean(axis=1)          # skipna, exactly as Part VII
s_tsmom_meta = pd.DataFrame(meta_legs).mean(axis=1)

# --- pairs: SPY-IVV z-score, graded at the next close, BOTH legs charged ----------
spread = (np.log(px["SPY"]) - np.log(px["IVV"])).dropna()
z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
raw = pd.Series(np.nan, index=z.index)
raw[z > 2], raw[z < -2] = -1.0, 1.0
raw[np.sign(z) != np.sign(z.shift(1))] = 0.0
base = raw.ffill().fillna(0.0)
two_legs = ((HS["SPY"] + COMM) + (HS["IVV"] + COMM)) * 1e-4   # a round trip is two trades
s_pairs = (base.shift(2) * spread.diff() - base.diff().abs().fillna(0.0) * two_legs).reindex(cal)

# --- xsmom: monthly sector ranks, weights held across daily returns ---------------
rsec = np.log(p4[SECT]).diff()
mp = p4[SECT].resample("ME").last()
ranks = mp.pct_change(11).shift(1).rank(axis=1)
w9 = ((ranks >= 7).astype(float) - (ranks <= 3).astype(float)) / 3
wd = w9.shift(1).reindex(cal, method="ffill")
s_xsmom = ((wd * rsec).sum(axis=1, min_count=len(SECT))
           - wd.diff().abs().fillna(0.0).sum(axis=1) * (HS["SEC"] + COMM) * 1e-4)

# --- shortvol: short 1m variance swap struck at prior month-end VIX, daily MTM ----
rs = np.log(px["SPY"]).diff().dropna()       # as Part IV: no leading NaN in the calendar
mkey = rs.index.to_period("M")
vix_me = p4["VIX"].resample("ME").last()
gate = ((p4["VIX3M"] > p4["VIX"]).where(p4["VIX3M"].notna())    # NaN, not False, pre-2006
        .resample("ME").last().shift(1))
k2 = (vix_me.shift(1) / 100) ** 2
k2.index, gate.index = k2.index.to_period("M"), gate.index.to_period("M")
iv = ((p4["VIX"] / 100) ** 2).reindex(rs.index)
pos_in, T = rs.groupby(mkey).cumcount() + 1, rs.groupby(mkey).transform("size")
rv = 252 * (rs ** 2).groupby(mkey).transform(lambda s: s.expanding().mean())
K2 = k2.reindex(mkey).to_numpy()
V = (pos_in / T) * (rv - K2) + (1 - pos_in / T) * (iv - K2)     # marked-to-market swap
V0 = V.groupby(mkey).shift(1).fillna(0.0)                       # V = 0 at inception
sv_raw = -(V - V0) * 100 * gate.reindex(mkey).to_numpy().astype(float)
K_SV = 0.10 / (np.sqrt(252) * sv_raw.std())                     # units convention, full sample
s_shortvol = (sv_raw * K_SV).reindex(cal)

# --- tom: SPY on the first three and the last trading day of each month ----------
pim = rs.groupby(mkey).cumcount() + 1
rev = rs[::-1].groupby(rs[::-1].index.to_period("M")).cumcount()[::-1] + 1
tmask = ((pim <= 3) | (rev == 1)).astype(float)
s_tom = (tmask * rs - tmask.diff().abs().fillna(0.0) * (HS["SPY"] + COMM) * 1e-4).reindex(cal)

# --- assemble --------------------------------------------------------------------
sleeves = pd.DataFrame({"s_tsmom": s_tsmom, "s_tsmom_meta": s_tsmom_meta,
                        "s_pairs": s_pairs, "s_xsmom": s_xsmom,
                        "s_shortvol": s_shortvol, "s_tom": s_tom})
assets = np.log(pd.concat([px[["SPY", "IVV", "TLT", "GLD"]],
                           p4[SECT + ["IWM", "EFA", "EEM"]]], axis=1, sort=False)).diff()
assets.columns = [f"r_{c}" for c in assets.columns]
ctx = p4[["VIX", "VIX3M"]].rename(columns={"VIX": "vix", "VIX3M": "vix3m"})
panel = pd.concat([sleeves, assets, ctx], axis=1, sort=False).astype("float64")
ret_cols = [c for c in panel.columns if c[:2] in ("s_", "r_")]
panel = panel.loc[panel[ret_cols].notna().any(axis=1)]
panel.index.name = "Date"
panel.to_parquet("data/part8.parquet")

names = [c for c in panel.columns if c.startswith("s_")]
pub = {"s_tsmom": "VII: +0.29", "s_tsmom_meta": "VII: +0.43", "s_pairs": "IV: -0.07",
       "s_xsmom": "IV: +0.06", "s_shortvol": "IV: monthly 1.55", "s_tom": "IV: +0.29"}
print(f"frozen: data/part8.parquet  {panel.shape[0]:,} rows x {panel.shape[1]} cols "
      f"({len(names)} sleeves), {panel.index[0].date()} -> {panel.index[-1].date()}")
for c in names:
    s = panel[c].dropna()
    print(f"  {c:13s} from {s.index[0].date()}  n {len(s):,}  "
          f"Sharpe {np.sqrt(252) * s.mean() / s.std():+.3f}  "
          f"vol {np.sqrt(252) * s.std():6.2%}  skew {s.skew():+6.2f}   published {pub[c]}")

sv_m = sv_raw.dropna().groupby(sv_raw.dropna().index.to_period("M")).sum()
p4sv = ((vix_me.shift(1) / 100) ** 2 - 252 * (rs ** 2).resample("ME").mean()) * 100
p4sv.index = p4sv.index.to_period("M")
p4sv = (p4sv * gate).dropna()
d = pd.concat([sv_m, p4sv], axis=1, sort=False).dropna()
print(f"shortvol telescopes to Part IV's monthly sleeve: "
      f"max abs diff {(d.iloc[:, 0] - d.iloc[:, 1]).abs().max():.1e}, "
      f"monthly Sharpe {np.sqrt(12) * p4sv.mean() / p4sv.std():.3f}")
print(f"pairs charged both legs: "
      f"{base.diff().abs().fillna(0.0).sum() * two_legs * 1e4 / (len(base) / 252):.0f} bp/yr "
      f"(Part IV's cost table: 25)")
# => frozen: data/part8.parquet  6,410 rows x 24 cols (6 sleeves), 2000-01-04 -> 2025-06-30
#      s_tsmom       from 2001-01-03  n 6,158  Sharpe +0.294  vol 12.23%  skew  -0.24   published VII: +0.29
#      s_tsmom_meta  from 2001-01-03  n 6,158  Sharpe +0.431  vol  8.22%  skew  -0.50   published VII: +0.43
#      s_pairs       from 2000-05-23  n 6,313  Sharpe -0.073  vol  0.96%  skew +38.77   published IV: -0.07
#      s_xsmom       from 2000-02-29  n 6,372  Sharpe +0.066  vol 14.89%  skew  -0.39   published IV: +0.06
#      s_shortvol    from 2006-08-01  n 4,758  Sharpe +1.471  vol 10.00%  skew  -5.65   published IV: monthly 1.55
#      s_tom         from 2000-01-04  n 6,410  Sharpe +0.286  vol  8.15%  skew  -1.23   published IV: +0.29
#    shortvol telescopes to Part IV's monthly sleeve: max abs diff 1.6e-15, monthly Sharpe 1.546
#    pairs charged both legs: 25 bp/yr (Part IV's cost table: 25)
```

Every sleeve lands on the number the course already published, which is the only reason the rest of the part is admissible. The variance-swap rebuild deserves its line: the daily marks sum to Part IV's monthly figure with a maximum absolute discrepancy of **1.6e-15** — machine epsilon, not agreement within tolerance. That is what "the same sleeve, differently sampled" is supposed to look like, and it is worth insisting on, because the daily path is now doing work the monthly series could not. A monthly Sharpe of 1.546 becomes a *daily* Sharpe of 1.471 attached to a skew of **−5.65** and a worst day of −12.9% in February 2018, and none of that tail was visible at monthly resolution.

Two facts about the panel govern how later lessons use it. It is **ragged on purpose** — no global `dropna` — because the common window across all six sleeves starts only in August 2006, and restricting to it changes verdicts (`xsmom` is +0.066 over its full history and negative on the common window). Each lesson states its own window; only the covariance work needs alignment. And `s_pairs` carries a skew of **+38.77**, which is not a strategy property: SPY and IVV, two funds tracking the same index, diverge by 3.77% in log terms on 2008-10-13 in the source data, and twenty days exceed a 1% divergence. That single print is worth 0.19 of book Sharpe, so `pairs` is excluded from every risk book below — a sleeve whose return distribution is dominated by one bad vendor mark cannot be used to teach tail measurement. It stays in the panel because [lesson three](03-risk-parity-diversification-factors.md) needs it to fail one more time.

## What volatility estimator, and how wrong is it

Every number in this lesson is downstream of a volatility estimate, so the estimate goes first. Three candidates, all standard: the 21-day close-to-close standard deviation, the RiskMetrics exponentially-weighted estimator at λ = 0.94, and Parkinson's range-based estimator, which extracts variance from the high–low range and therefore uses information a close-to-close estimate throws away. The honest test is not how well each fits the past but how well it predicts the volatility that actually arrives — realized vol over the *next* 21 days:

```python
import numpy as np
import pandas as pd

bars = pd.read_parquet("data/part5.parquet").xs("SPY", axis=1, level=1).dropna()
c, h, l = bars["Close"], bars["High"], bars["Low"]
r = np.log(c).diff()
fwd = np.sqrt(252) * r.shift(-1).rolling(21).std().shift(-20)      # the next 21 days

est = {"close-to-close 21d": np.sqrt(252) * r.rolling(21).std(),
       "EWMA lambda 0.94": np.sqrt(252) * np.sqrt((r ** 2).ewm(alpha=1 - 0.94).mean()),
       "Parkinson 21d": np.sqrt(252 * (np.log(h / l) ** 2
                                       / (4 * np.log(2))).rolling(21).mean())}
print(f"target = realized vol over the next 21 days, mean {fwd.mean():.2%}")
for k, v in est.items():
    d = pd.concat([v.rename("e"), fwd.rename("f")], axis=1).dropna()
    slope = np.polyfit(np.log(d.e), np.log(d.f), 1)[0]
    print(f"  {k:20s} corr {d.e.corr(d.f):.4f}  mean {d.e.mean():.2%}  "
          f"bias {d.e.mean() - d.f.mean():+.2%}  MAE {(d.e - d.f).abs().mean():.3%}  "
          f"log-log slope {slope:.3f}")
# => target = realized vol over the next 21 days, mean 16.45%
#      close-to-close 21d   corr 0.6437  mean 16.46%  bias +0.04%  MAE 5.598%  log-log slope 0.645
#      EWMA lambda 0.94     corr 0.6644  mean 16.81%  bias +0.36%  MAE 5.245%  log-log slope 0.737
#      Parkinson 21d        corr 0.6715  mean 13.76%  bias -2.66%  MAE 4.982%  log-log slope 0.737
```

The ranking is not the same on every column, and that is the point. Parkinson wins on correlation (0.672) and on mean absolute error (4.98%) while being by far the *most biased* — it runs 2.66 percentage points low, because the high–low range of a session cannot see the overnight gap, and for an ETF that gaps on macro news the missing piece is systematic rather than random. Close-to-close is nearly unbiased (+0.04%) and the least informative — that a biased estimator wins on error, and that the ranking inverts once three of them are averaged, is [Bias and Variance](../appendix/part-10-statistics-foundations/07-bias-and-variance.md). The practical resolution is the one professional risk systems actually use: take the range estimator's *shape* and correct its *level*, or use EWMA, which here buys most of Parkinson's accuracy with a seventh of its bias.

The column that matters more than the ranking is the last one. All three log-log slopes sit between 0.645 and 0.737, comfortably below one, which says that a doubling of today's estimated volatility predicts substantially *less* than a doubling of tomorrow's — volatility mean-reverts ([Part III](../part-03-statistics/03-time-series.md) fitted its lag structure), so every estimator over-extrapolates, and a risk report that reads today's number as a forecast will be too frightened in crises and too relaxed in calm. Shrinking toward the long-run mean is not conservatism; it is the regression coefficient.

## Value at Risk, three ways, and the backtest that fails them

Value at Risk answers one question — *what loss will not be exceeded on 99 days out of 100* — and the three standard ways of answering it differ only in what they assume about the shape of the distribution. Parametric VaR assumes a normal; historical VaR assumes the past sample is the distribution; Monte Carlo assumes whatever you sample from ([Value at Risk](../appendix/part-18-quant-finance-applications/10-value-at-risk.md) derives all three). Since the answers are testable, the interesting part is not the levels but the backtest: Kupiec's proportion-of-failures test asks whether the *number* of breaches matches the promise, and Christoffersen's independence test asks whether the breaches are *scattered* the way independent events should be. Both run on the book — the five surviving sleeves, each scaled to the same standalone volatility:

```python
import numpy as np
import pandas as pd
from scipy import stats

p8 = pd.read_parquet("data/part8.parquet")
S = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]
w = p8[S].dropna()
book = (w / (np.sqrt(252) * w.std()) * 0.10 / len(S)).sum(axis=1)   # each sleeve at 2% vol

def kupiec(x, n, p):                       # proportion of failures, chi2(1)
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - x / n) + x * np.log(x / n)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1)

def christoffersen(br):                    # breach independence, chi2(1)
    b = br.astype(int).values
    n00 = ((b[:-1] == 0) & (b[1:] == 0)).sum(); n01 = ((b[:-1] == 0) & (b[1:] == 1)).sum()
    n10 = ((b[:-1] == 1) & (b[1:] == 0)).sum(); n11 = ((b[:-1] == 1) & (b[1:] == 1)).sum()
    pi = (n01 + n11) / len(b[:-1]); p0 = n01 / (n00 + n01); p1 = n11 / (n10 + n11)
    lg = lambda k, q: k * np.log(q) if k > 0 and q > 0 else 0.0
    ll0 = lg(n00 + n10, 1 - pi) + lg(n01 + n11, pi)
    ll1 = lg(n00, 1 - p0) + lg(n01, p0) + lg(n10, 1 - p1) + lg(n11, p1)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1), n11

rng = np.random.default_rng(0)
n, a = len(book), 0.01
print(f"the book: n {n:,}, {book.index[0].date()} to {book.index[-1].date()}, "
      f"expected breaches at 99% = {n * a:.1f}")
fixed = {"parametric-normal": book.mean() + book.std() * stats.norm.ppf(a),
         "historical": book.quantile(a),
         "MC (bootstrap)": np.quantile(rng.choice(book.values, 1_000_000), a)}
for k, v in fixed.items():
    br = book < v
    p_ind, n11 = christoffersen(br)
    print(f"  {k:18s} VaR {-v:.3%}  breaches {br.sum():3d} ({br.sum() / (n * a):.2f}x)  "
          f"Kupiec p {kupiec(br.sum(), n, a):.2e}  indep p {p_ind:.4f}  n11 {n11}")

ewma = np.sqrt((book ** 2).ewm(alpha=1 - 0.94).mean()).shift(1)     # conditional VaR
br = book < ewma * stats.norm.ppf(a)
p_ind, n11 = christoffersen(br)
print(f"  {'EWMA conditional':18s} time-varying   breaches {br.sum():3d} "
      f"({br.sum() / (n * a):.2f}x)  Kupiec p {kupiec(br.sum(), n, a):.2e}  "
      f"indep p {p_ind:.4f}  n11 {n11}")
byr = (book < fixed["parametric-normal"]).groupby(book.index.year).sum().sort_values()
print(f"  normal-VaR breaches by year, worst four: "
      f"{', '.join(f'{y}: {v}' for y, v in byr.tail(4).items())}")
# => the book: n 4,758, 2006-08-01 to 2025-06-30, expected breaches at 99% = 47.6
#      parametric-normal  VaR 0.825%  breaches 119 (2.50x)  Kupiec p 0.00e+00  indep p 0.0000  n11 12
#      historical         VaR 1.121%  breaches  48 (1.01x)  Kupiec p 9.51e-01  indep p 0.0013  n11 4
#      MC (bootstrap)     VaR 1.119%  breaches  48 (1.01x)  Kupiec p 9.51e-01  indep p 0.0013  n11 4
#      EWMA conditional   time-varying   breaches 117 (2.46x)  Kupiec p 0.00e+00  indep p 0.5208  n11 4
#      normal-VaR breaches by year, worst four: 2009: 10, 2018: 11, 2020: 14, 2008: 17
```

The parametric number is not approximately wrong; it is wrong by a factor of two and a half, with a Kupiec p-value that underflows to zero. A model promising 48 breaches in nineteen years delivered **119**, and seventeen of them arrived in 2008 alone — more in one year than the model budgeted for a decade. The diagnosis is entirely the normal assumption: the book's kurtosis is 9.8, so the 2.33-sigma point that *should* cut off 1% of a Gaussian cuts off 2.5% of this distribution.

The historical and bootstrap methods fix the count and pass Kupiec comfortably — 48 breaches apiece against 47.6 expected — and then fail the other test. Their independence p-value of 0.0013 says the breaches arrive in clusters, and the year table names the clusters. A static quantile of the whole sample is a single unconditional number applied to a market that alternates between calm and turmoil; it is too loose most of the time and too tight exactly when it matters. It is worth being clear about what kind of object is being compared here: a VaR is a value of $F^{-1}$ and nothing more, so on a finite sample its leading digit is an order statistic and its trailing digits are an interpolation convention — [Cumulative Distribution Functions](../appendix/part-03-random-variables/02-cumulative-distribution-functions.md) shows three standard conventions disagreeing by a factor of 1.86 on a single year of returns. **Getting the level right does not get the timing right**, and a risk limit that is correct on average while being wrong for six consecutive weeks is not a risk limit.

The last row is the one worth remembering, because it is the fix that isn't. Conditioning the VaR on an EWMA of recent volatility — the RiskMetrics construction, and the standard response to clustering — works exactly as advertised on the problem it targets: the independence p-value jumps from 0.0013 to **0.5208** and the consecutive-breach count falls to four, so the clustering is genuinely gone. The count is no better. **117 breaches against 48 expected**, because conditioning rescales the distribution day by day without changing its shape, and the shape was the problem. Fat tails do not become thin when you divide by the right sigma. Fixing the timing and fixing the level are two separate repairs, and the standard tool performs one of them.

## Expected shortfall and the shape of the tail

VaR names a threshold and says nothing about what lies beyond it — a distribution with a 1.1% VaR and a −4% worst day and one with a 1.1% VaR and a −40% worst day are indistinguishable to it. Expected shortfall, the average loss *conditional on* exceeding the threshold, looks past the cutoff, and it is [coherent](../appendix/part-18-quant-finance-applications/11-expected-shortfall.md) where VaR is not: the ES of a combined book can never exceed the sum of its parts', a guarantee VaR cannot make. It is also the measure Basel's FRTB regime moved to, swapping a 99% VaR for a 97.5% ES on the grounds that the two are nearly identical under a normal:

```python
import numpy as np
import pandas as pd
from scipy import stats

p8 = pd.read_parquet("data/part8.parquet")
def book(cols, target=0.10):
    w = p8[cols].dropna()
    return (w / (np.sqrt(252) * w.std()) * target / len(cols)).sum(axis=1)

S = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]
g = stats.norm.expect(lambda x: x, lb=-np.inf, ub=stats.norm.ppf(0.025)) / 0.025
for lab, cols in [("the book", S), ("surviving", ["s_tsmom", "s_shortvol"])]:
    b = book(cols)
    v975, v99 = b.quantile(0.025), b.quantile(0.01)
    es = b[b <= v975].mean()
    print(f"  {lab:10s} VaR97.5 {-v975:.3%}  ES97.5 {-es:.3%}  VaR99 {-v99:.3%}   "
          f"ES/VaR at 97.5 {es / v975:.3f} (Gaussian {g / stats.norm.ppf(0.025):.3f})   "
          f"ES97.5/VaR99 {es / v99:.3f}   kurtosis {b.kurtosis():.1f}")
# =>   the book   VaR97.5 0.822%  ES97.5 1.200%  VaR99 1.121%   ES/VaR at 97.5 1.459 (Gaussian 1.193)   ES97.5/VaR99 1.070   kurtosis 9.8
#      surviving  VaR97.5 0.945%  ES97.5 1.638%  VaR99 1.471%   ES/VaR at 97.5 1.734 (Gaussian 1.193)   ES97.5/VaR99 1.114   kurtosis 29.2
```

Under a normal distribution the average loss beyond the 97.5% point is 1.193 times the threshold itself. The book runs at **1.459** and the two-sleeve surviving book at **1.734** — 45% more tail than normality allows. The ratio earns its place as a diagnostic precisely because it is unit-free and leverage-free: it does not care how large the book is, only how far the losses that get through the threshold travel past it, and a reading near 1.2 means Gaussian intuition is safe while a reading near 1.75 means it is not. The Basel substitution survives on this data — ES97.5 is 1.070 and 1.114 times VaR99, so the regulatory swap is conservative by 7 to 11% rather than the 0.5% a Gaussian would predict.

The comparison between the two books is the more instructive half. The *better* book is the *fatter-tailed* one: concentrating into `tsmom` and `shortvol` lifts the Sharpe from 0.827 to 1.174 while lifting kurtosis from 9.8 to **29.2** and the ES ratio from 1.459 to 1.734. That is not a coincidence to be noted and passed over — it is the trade the surviving book actually made. `shortvol` earns its Sharpe by selling insurance, and insurance premiums are collected in small regular amounts and repaid in rare large ones. Every risk statistic in this part will keep rediscovering that the highest-Sharpe construction available to this course is also the one whose left tail is least well described by its own summary numbers.

## Whose risk is it: marginal and component contributions

A book's risk is not distributed like its capital. Each position's *marginal* contribution is the derivative of portfolio volatility with respect to its weight, and multiplying by the weight gives the *component* contribution, a decomposition with the useful property that the pieces sum exactly to the total:

$$
\sigma_p \;=\; \sum_i w_i \, \frac{(\Sigma w)_i}{\sigma_p}, \qquad
\text{CRC}_i \;=\; w_i \, \frac{(\Sigma w)_i}{\sigma_p} .
$$

Two allocations, deliberately: equal dollars across all six sleeves, and equal *volatility* across the five that survive. The second is the interesting one, because equalizing volatility removes size as an explanation and leaves correlation as the only thing that can create dispersion:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")

def decompose(cols, weights, label):
    w = p8[cols].dropna()
    cov = (w * weights).cov().values * 252
    v = np.ones(len(cols))
    pv = np.sqrt(v @ cov @ v)
    crc = v * ((cov @ v) / pv)
    print(f"{label}: book ann vol {pv:.2%}, diversification ratio "
          f"{np.sqrt(np.diag(cov)).sum() / pv:.2f}, equal share would be {1 / len(cols):.1%}")
    for i, k in enumerate(cols):
        print(f"    {k:13s} standalone {np.sqrt(252) * w[k].std():6.2%}   "
              f"MRC {(cov @ v)[i] / pv:6.2%}   share of book risk {crc[i] / pv:6.1%}")

S6 = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom", "s_pairs"]
S5 = S6[:5]
decompose(S6, 1 / 6, "equal DOLLARS, six sleeves")
w = p8[S5].dropna()
decompose(S5, 0.10 / (np.sqrt(252) * w.std()) / 5, "equal VOLATILITY, five sleeves")
print(f"  corr(tsmom, tsmom_meta) = {w.s_tsmom.corr(w.s_tsmom_meta):.3f};  "
      f"xsmom mean corr to the rest = {w.corr()['s_xsmom'].drop('s_xsmom').mean():+.3f}")
# => equal DOLLARS, six sleeves: book ann vol 4.99%, diversification ratio 1.77, equal share would be 16.7%
#        s_tsmom       standalone 10.46%   MRC  1.22%   share of book risk  24.5%
#        s_tsmom_meta  standalone  7.58%   MRC  0.85%   share of book risk  17.0%
#        s_xsmom       standalone 15.63%   MRC  1.72%   share of book risk  34.4%
#        s_shortvol    standalone 10.00%   MRC  0.75%   share of book risk  15.0%
#        s_tom         standalone  8.16%   MRC  0.46%   share of book risk   9.2%
#        s_pairs       standalone  1.04%   MRC  0.00%   share of book risk   0.0%
#    equal VOLATILITY, five sleeves: book ann vol 5.76%, diversification ratio 1.74, equal share would be 20.0%
#        s_tsmom       standalone 10.46%   MRC  1.45%   share of book risk  25.1%
#        s_tsmom_meta  standalone  7.58%   MRC  1.48%   share of book risk  25.6%
#        s_xsmom       standalone 15.63%   MRC  0.99%   share of book risk  17.3%
#        s_shortvol    standalone 10.00%   MRC  0.98%   share of book risk  17.0%
#        s_tom         standalone  8.16%   MRC  0.86%   share of book risk  14.9%
#      corr(tsmom, tsmom_meta) = 0.756;  xsmom mean corr to the rest = +0.108
```

The equal-dollar table is the naive book and it is badly out of balance: `xsmom` takes **34.4%** of the risk against a 16.7% share of the capital, and `pairs` — sized identically in dollars — contributes **0.0%**, its 1.04% volatility rounding away to nothing beside sleeves ten times its size. An equal-weight book is not a neutral book; it is a bet on whichever sleeve happens to be most volatile, placed by default. This is the same diagnosis [Part IV](../part-04-strategy-development/06-position-sizing-and-risk-budgeting.md) reached on three trend sleeves, arriving again at six and worse.

Equalizing volatility is the obvious repair and it does not work either, which is the section's real finding. Every sleeve now runs at exactly 2% standalone volatility, so size cannot explain anything — and the shares still range from **14.9% to 25.6%**. What remains is correlation, and the correlation doing the damage is named on the last line: `tsmom` and `tsmom_meta` correlate at **0.756**, because they are the same strategy, one of them filtered. Two positions that move together are, for risk purposes, one larger position, so the pair jointly consumes 50.7% of a five-sleeve book. Note which one comes out on top: `tsmom_meta` has the **lowest standalone volatility of the five (7.58%)** and the **largest share of book risk (25.6%)**. Standalone volatility is not a risk budget, is not a good proxy for one, and ranks these sleeves in close to the wrong order.

The diversification ratio — the sum of standalone volatilities over the book's actual volatility — puts one number on what the correlations are worth: **1.74**, meaning the book runs at 57% of the risk its parts would carry if they moved in lockstep. That is a real and substantial dividend. It is also the number [lesson three](03-risk-parity-diversification-factors.md) will show is far more fragile than it looks, and [lesson four](04-portfolio-optimization-and-correlation.md) will show can collapse precisely when it is needed.

## Aggregation: horizons, and two errors that do not cancel

Risk is quoted daily and experienced over weeks, so every risk system scales, and almost all of them scale by $\sqrt{h}$. That rule is exactly right for independent increments and wrong here in two different directions at once:

```python
import numpy as np
import pandas as pd
from scipy import stats

p8 = pd.read_parquet("data/part8.parquet")
S = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]
w = p8[S].dropna()
book = (w / (np.sqrt(252) * w.std()) * 0.10 / len(S)).sum(axis=1)

print(f"lag-1 autocorrelation {book.autocorr(1):+.3f}, "
      f"sum of lags 1-5 {sum(book.autocorr(i) for i in range(1, 6)):+.3f}")
v1 = book.mean() + book.std() * stats.norm.ppf(0.01)
for h in [5, 10, 21]:
    act = book.rolling(h).sum().dropna()
    print(f"  h={h:2d}:  vol  sqrt-h {np.sqrt(h) * book.std():.3%} vs actual {act.std():.3%} "
          f"({np.sqrt(h) * book.std() / act.std() - 1:+.1%})    "
          f"VaR99  sqrt-h normal {-np.sqrt(h) * v1:.3%} vs actual historical "
          f"{-act.quantile(0.01):.3%} ({np.sqrt(h) * v1 / act.quantile(0.01) - 1:+.1%})")
# => lag-1 autocorrelation -0.009, sum of lags 1-5 -0.100
#      h= 5:  vol  sqrt-h 0.811% vs actual 0.779% (+4.1%)    VaR99  sqrt-h normal 1.844% vs actual historical 2.227% (-17.2%)
#      h=10:  vol  sqrt-h 1.147% vs actual 1.054% (+8.8%)    VaR99  sqrt-h normal 2.608% vs actual historical 2.950% (-11.6%)
#      h=21:  vol  sqrt-h 1.662% vs actual 1.497% (+11.0%)    VaR99  sqrt-h normal 3.780% vs actual historical 3.948% (-4.3%)
```

Scaled *volatility* is too high at every horizon, by 4% at a week and **11% at a month**, and the reason is on the first line: the book's autocorrelations sum to −0.100 over the first five lags, so it mildly mean-reverts, and mean-reverting increments accumulate less variance than independent ones. Scaled *VaR* is too low at every horizon, by **17% at a week**. Two errors, opposite signs — and they do not cancel, they compound into a false sense of control, because the conservative one lands on the number nobody breaches and the aggressive one on the number that triggers action. The mechanism is that $\sqrt{h}$ scaling of a *normal* VaR carries the normality error along with it, and over five days that tail error is larger than the mean-reversion credit. A monthly risk limit derived by scaling a daily one understates the loss it exists to prevent by roughly a fifth, and the fix is unglamorous: measure the h-day distribution at h days.

## The one-page report

All of it collapses onto the artifact a portfolio manager actually reads before the open. The rule for this page is that every line is either a number someone can act on or a number that says how much to distrust the others:

```python
import numpy as np
import pandas as pd

p8 = pd.read_parquet("data/part8.parquet")
S = ["s_tsmom", "s_tsmom_meta", "s_xsmom", "s_shortvol", "s_tom"]
w = p8[S].dropna()
sc = 0.10 / (np.sqrt(252) * w.std()) / len(S)
book = (w * sc).sum(axis=1)
cov = (w * sc).cov().values * 252
v = np.ones(len(S))
pv = np.sqrt(v @ cov @ v)
crc = v * ((cov @ v) / pv)

asof = book.index[-1]
ewma = np.sqrt(252 * (book ** 2).ewm(alpha=1 - 0.94).mean().iloc[-1])
var99, es = book.quantile(0.01), book[book <= book.quantile(0.025)].mean()
ytd = book[book.index.year == asof.year]
eq = np.exp(book.cumsum())
dd = eq / eq.cummax() - 1

print(f"{'DAILY RISK REPORT':<34}{asof:%Y-%m-%d}")
print(f"  ex-ante vol   EWMA {ewma:>7.2%}   full-sample {np.sqrt(252) * book.std():.2%}")
print(f"  VaR99 / ES97.5      {-var99:>7.2%} / {-es:.2%}   "
      f"(1-day, historical; scale to h days at h days)")
print(f"  drawdown      current {dd.iloc[-1]:>6.2%}   worst ever {dd.min():.2%}")
print(f"  YTD           return {np.exp(ytd.sum()) - 1:>+7.2%}   VaR breaches "
      f"{(ytd < var99).sum()} in {len(ytd)} days")
print(f"  risk budget   largest contributor {S[int(np.argmax(crc))]} "
      f"{crc.max() / pv:.1%} of {pv:.2%}")
for i, k in enumerate(S):
    s = w[k].iloc[-252:]
    print(f"      {k:13s} share {crc[i] / pv:>5.1%}   "
          f"trailing-252d Sharpe {np.sqrt(252) * s.mean() / s.std():+.2f}")
# => DAILY RISK REPORT                 2025-06-30
#      ex-ante vol   EWMA   3.40%   full-sample 5.76%
#      VaR99 / ES97.5        1.12% / 1.20%   (1-day, historical; scale to h days at h days)
#      drawdown      current  0.00%   worst ever -11.22%
#      YTD           return  +2.96%   VaR breaches 2 in 122 days
#      risk budget   largest contributor s_tsmom_meta 25.6% of 5.76%
#          s_tsmom       share 25.1%   trailing-252d Sharpe +0.55
#          s_tsmom_meta  share 25.6%   trailing-252d Sharpe +1.25
#          s_xsmom       share 17.3%   trailing-252d Sharpe +0.48
#          s_shortvol    share 17.0%   trailing-252d Sharpe +0.61
#          s_tom         share 14.9%   trailing-252d Sharpe -0.57
```

Read it the way a risk committee would. The EWMA volatility of 3.40% against a full-sample 5.76% says the book is currently running at 59% of its normal risk, which is information about the *regime*, not about the positions — nothing was cut, the market went quiet, and the honest gloss from two sections ago is that this reading will lag the next regime break rather than anticipate it. Two breaches in 122 days is 1.6% against a 1% budget: unremarkable alone, worth watching if it persists. The risk-budget block is where the decisions live. The two trend sleeves jointly own **50.7%** of a book that holds five, and the report says so on its face rather than burying it inside an equal-looking weight vector. And `tom`, at a trailing Sharpe of −0.57, is spending 14.9% of the risk budget to lose money slowly — a fact the report surfaces but cannot adjudicate, because one year of Sharpe is one year of noise. Lesson three will adjudicate it properly, with a t-statistic.

!!! warning "A risk model that has never been backtested is a number with a decimal point, not a measurement"
    The parametric VaR in this lesson failed at odds that underflow double precision, and it is the single most widely deployed risk number in finance. It failed for a reason knowable in advance — the book's kurtosis is 9.8 and the model assumes 3 — and the failure was invisible to anyone who computed the number without ever checking it against outcomes. The discipline is not preferring one estimator to another; it is that every risk figure a desk publishes carries an implicit forecast, forecasts can be scored, and the scoring is cheap. Count the breaches. Test whether they cluster. Do it on a schedule, not after the loss that prompts the question.

!!! abstract "Key takeaways"
    - The part runs off one frozen panel, `data/part8.parquet` (6,410 rows × 24 columns), whose six sleeves each reproduce their published Part IV or Part VII Sharpe — including a daily variance swap that telescopes to Part IV's monthly `shortvol` to within 1.6e-15.
    - Volatility estimators trade bias against noise: Parkinson has the best correlation to forward realized vol (0.672) and the worst bias (−2.66%, the overnight gap it cannot see); all three have log-log slopes of 0.65–0.74, so every one of them over-extrapolates today's level.
    - Parametric normal VaR delivered **119 breaches against 48 expected** (2.50×, Kupiec p underflowing to zero), seventeen of them in 2008 alone — the book's kurtosis is 9.8, so the Gaussian 2.33-sigma point cuts off 2.5% of this distribution rather than 1%.
    - Historical VaR fixes the count (48 breaches, Kupiec p 0.95) and fails independence (p 0.0013); EWMA-conditional VaR fixes the clustering (independence p **0.5208**) and not the count (117 breaches). Level and timing are separate repairs and the standard tool performs one.
    - Expected shortfall at 97.5% is 1.459× its own VaR for the book and **1.734×** for the surviving book, against 1.193 under a normal — and the higher-Sharpe book is the fatter-tailed one, because its edge is sold insurance.
    - Equal dollars hands `xsmom` 34.4% of book risk and `pairs` 0.0%; equalizing *volatility* still leaves shares spanning 14.9% to 25.6%, because `tsmom` and `tsmom_meta` correlate at 0.756 and jointly consume half the book. The lowest-volatility sleeve owns the largest risk share.
    - √h scaling overstates monthly volatility by 11% (the book mean-reverts, lags 1–5 sum to −0.100) while understating 5-day VaR99 by 17% — two errors of opposite sign that compound rather than cancel.

## Where this goes next

The book has now been measured, and measurement is what makes sizing arguable rather than arbitrary: a 5.76% volatility, a 1.12% daily VaR, a −11.2% worst drawdown and a diversification ratio of 1.74 are the raw material of the question every allocator eventually asks, which is *why this much and not twice as much*. [Kelly, Volatility Targeting, and Leverage](02-kelly-vol-targeting-leverage.md) answers it — deriving the growth-optimal bet, showing that the growth-optimal bet on the best book in this course is arithmetically impossible to place, pricing the financing that leverage actually costs, and testing whether cutting size in a drawdown protects the book or merely locks in the loss. The answer to that last one depends on which book you ask, which is why it needs two.
