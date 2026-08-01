# Alternative Data

[Part VII's feature-engineering lesson](../part-07-machine-learning/01-feature-engineering-for-ml.md) built nineteen features from prices alone and measured what they were worth: an AUC of 0.515, thin enough that a single round of boosting overfitted it. Its closing argument was that more *rows* and more *kinds* of information are the honest way out of that corner, and it pointed here. This module is about the kinds — news text, regulatory filings, satellite imagery, geolocation, card transactions — and about the discipline that decides whether any of them is worth its invoice.

That discipline is mostly statistics, not natural language processing. The engineering of an alternative dataset is usually tractable; what kills projects is that the effects are small, the samples are short, the histories are frequently contaminated by the vendor's own hindsight, and the timestamps are wrong in ways that flatter the backtest. This module quantifies each of those. It fetches two real Apple 10-K filings from EDGAR and measures how little they change year to year; it plants a known 25-basis-point event effect and shows what sample size is needed to *measure* rather than merely *detect* it; it builds a backfilled vendor history and catches it with a two-sample test; it prices a one-day timestamp error at **5.46 Sharpe points**; and it ends with the arithmetic that turns a vendor's pitch deck into a purchase decision, where the same dataset is obviously worth buying at a billion dollars of assets and obviously not at fifty million. [Part II's SQL lesson](../part-02-python/05-sql-and-data-storage.md) already established point-in-time correctness as a database discipline; this module is what happens when you put a price on violating it.

## An alternative dataset is a causal hypothesis with an invoice

The taxonomy that matters is not by source but by the length of the causal chain between the observation and the cash flow. **Short chains** are the good ones: card-transaction panels aggregate to a retailer's revenue almost directly, and app-download counts map to a subscription business's user growth with few intervening steps. **Long chains** are where most disappointment lives: satellite images of parking lots relate to same-store sales, which relate to segment revenue, which relates to consolidated earnings, which relates to the surprise against consensus, which relates to the stock's reaction — five links, each with its own noise, each of which must be estimated. [Part I's canonical example](../part-01-foundations/01-what-is-algorithmic-trading.md) was exactly this, and its verdict — expensive to build, quick to decay — is what the arithmetic below explains.

Three questions decide whether a dataset is worth evaluating at all, and they can be asked before any data changes hands. *What is the chain, and how many links does it have?* *Is the history point-in-time, or reconstructed?* *How many independent observations does the history actually contain* — not rows, but independent bets, which for a quarterly signal on 100 names over five years is 2,000 at the most generous count and often far fewer once you notice that the names move together. That last number, not the terabyte count, determines what can be learned.

## Filings barely change, which is exactly why the changes matter

Regulatory filings are the most accessible alternative dataset in existence: complete, free, timestamped by the regulator, and impossible to backfill because the filing date is a matter of public record. The research finding that made them interesting — Cohen, Malloy, and Nguyen's "Lazy Prices" — is that companies recycle language year over year, so the *changes* are rare, deliberate, and informative, and the market underreacts to them. Measure that recycling directly:

```python
# one-time download — requires a network connection
import re
import time
import numpy as np
import requests
from bs4 import BeautifulSoup

HEAD = {"User-Agent": "quant-course-example ianvs2014@gmail.com"}   # SEC requires a real contact

def filing_text(cik, accession, doc):
    acc = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    html = requests.get(url, headers=HEAD, timeout=60).text
    return re.sub(r"\s+", " ", BeautifulSoup(html, "html.parser").get_text(" "))

def risk_section(text):                       # Item 1A, between its heading and Item 1B
    low = text.lower()
    starts = [m.start() for m in re.finditer(r"item\s*1a\.?\s*risk factors", low)]
    ends = [m.start() for m in re.finditer(r"item\s*1b\.?\s*unresolved", low)]
    s = starts[-1]
    return text[s:[x for x in ends if x > s][0]]

filings = [("2023-11-03", "0000320193-23-000106", "aapl-20230930.htm"),
           ("2024-11-01", "0000320193-24-000123", "aapl-20240928.htm"),
           ("2025-10-31", "0000320193-25-000079", "aapl-20250927.htm")]
secs = {}
for date, acc, doc in filings:
    secs[date] = risk_section(filing_text("0000320193", acc, doc))
    print(f"{date}: Item 1A is {len(secs[date]):,} characters")
    time.sleep(0.5)                           # be polite to EDGAR

STOP = set("the a an and or of to in for on with is are was were be been by as at that this it "
           "its from we our us may can could would will not no if than then such other which "
           "their there these those have has had do does did but also".split())

def term_freq(text):
    out = {}
    for w in re.findall(r"[a-z']+", text.lower()):
        if len(w) > 2 and w not in STOP:
            out[w] = out.get(w, 0) + 1
    return out

def cosine(a, b):
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0) for k in keys], float)
    vb = np.array([b.get(k, 0) for k in keys], float)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

dates = [d for d, _, _ in filings]
for i in range(len(dates) - 1):
    a, b = secs[dates[i]], secs[dates[i + 1]]
    print(f"{dates[i]} -> {dates[i + 1]}: cosine similarity {cosine(term_freq(a), term_freq(b)):.4f}, "
          f"length change {(len(b) / len(a) - 1):+.1%}")
# => 2023-11-03: Item 1A is 67,876 characters
#    2024-11-01: Item 1A is 68,759 characters
#    2025-10-31: Item 1A is 68,045 characters
#    2023-11-03 -> 2024-11-01: cosine similarity 0.9977, length change +1.3%
#    2024-11-01 -> 2025-10-31: cosine similarity 0.9894, length change -1.0%
```

Two consecutive risk-factor sections are **99.77% and 98.94% identical** by term frequency, and their lengths move by about a percent. Companies are not rewriting their risk disclosures; they are editing them, and the edits are what a research process should isolate. Notice also what this dataset offers that most vendors cannot: the filing date is the knowledge date, stamped by a third party, with no possibility of silent revision. That property is worth more than most of the fancier datasets in this module, and it is free.

The natural next step is sentiment scoring, and the natural mistake is to reach for a general-purpose lexicon. Loughran and McDonald's central finding is that words like "liability," "cost," and "capital" carry negative sentiment in ordinary English and are neutral accounting vocabulary in filings, so a general lexicon mostly measures how much accounting a document contains. Domain-specific word lists exist for exactly this reason, and the correct instinct — before any modeling — is to ask what a score is actually counting.

## An event study is a microscope with a computable resolution

Most alternative-data claims are event claims: after *this* happens, the stock does *that*. The standard instrument is the event study. Estimate a market model on a clean window before the event, $r_{it} = \alpha_i + \beta_i r_{mt} + \varepsilon_{it}$, then measure the abnormal return on the event day and cumulate over a window:

$$
AR_{it} \;=\; r_{it} - \bigl(\hat\alpha_i + \hat\beta_i r_{mt}\bigr),
\qquad
CAR_i(\tau_1, \tau_2) \;=\; \sum_{t=\tau_1}^{\tau_2} AR_{it}.
$$

The test statistic across $N$ events is $t = \overline{CAR}/(\hat\sigma_{CAR}/\sqrt{N})$, and rearranging it gives the number every researcher should compute *before* collecting data — the minimum effect the study could detect at all:

$$
\text{minimum detectable effect} \;\approx\; \frac{2\,\hat\sigma_{AR}\sqrt{\tau_2 - \tau_1 + 1}}{\sqrt{N}} .
$$

Plant a known 25-basis-point effect on randomly chosen dates and watch what different sample sizes can say about it:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
r = np.log(px[["SPY", "TLT"]]).diff().dropna()
mkt, asset = r["SPY"].values, r["TLT"].values

def event_study(n_events, effect_bp, seed=7):
    rng = np.random.default_rng(seed)
    dates = rng.choice(np.arange(300, len(r) - 30), n_events, replace=False)
    y = asset.copy()
    for d in dates:
        y[d] += effect_bp / 1e4                      # plant the effect
    ars, sds = [], []
    for d in dates:
        est = slice(d - 250, d - 20)                 # clean estimation window
        X = np.column_stack([np.ones(230), mkt[est]])
        beta, *_ = np.linalg.lstsq(X, y[est], rcond=None)
        sds.append((y[est] - X @ beta).std(ddof=2) * 1e4)
        ars.append((y[d] - (beta[0] + beta[1] * mkt[d])) * 1e4)
    ars = np.array(ars)
    return ars.mean(), ars.std(ddof=1) / np.sqrt(len(ars)), np.mean(sds)

for n in [500, 60, 12]:
    m, se, sd_ar = event_study(n, 25.0)
    print(f"  N = {n:>3} events: mean AR {m:+6.1f} bp, 95% CI [{m - 1.96 * se:5.1f}, "
          f"{m + 1.96 * se:5.1f}], t = {m / se:.2f}   (planted 25.0 bp)")
m0, se0, sd_ar = event_study(60, 0.0)
print(f"  N =  60, nothing planted: mean AR {m0:+.1f} bp, t = {m0 / se0:+.2f}")
print(f"  daily abnormal-return sd is {sd_ar:.0f} bp, so the minimum detectable effect is:")
for n in [12, 60, 500, 5000]:
    print(f"     N = {n:>4} events: {2 * sd_ar / np.sqrt(n):5.1f} bp")
# =>   N = 500 events: mean AR  +24.7 bp, 95% CI [ 17.4,  32.0], t = 6.67   (planted 25.0 bp)
#      N =  60 events: mean AR  +31.2 bp, 95% CI [  9.2,  53.2], t = 2.78   (planted 25.0 bp)
#      N =  12 events: mean AR  +32.1 bp, 95% CI [  0.6,  63.5], t = 2.00   (planted 25.0 bp)
#      N =  60, nothing planted: mean AR +6.5 bp, t = +0.57
#      daily abnormal-return sd is 84 bp, so the minimum detectable effect is:
#         N =   12 events:  48.5 bp
#         N =   60 events:  21.7 bp
#         N =  500 events:   7.5 bp
#         N = 5000 events:   2.4 bp
```

Read the confidence intervals rather than the p-values. At 500 events the study *measures* the effect: 24.7 bp against a planted 25.0, with an interval narrow enough to act on. At 60 events it *detects* something (t = 2.78) while the interval spans 9 to 53 bp — a range across which the economics change completely. At 12 events the point estimate of 32.1 bp is off by 29%, the interval runs from 0.6 to 63.5, and t = 2.00 would be reported as significant in most write-ups. Detection is not measurement, and a "significant" result whose interval spans an order of magnitude cannot size a position.

The bottom table is the honest planning tool, and it should be computed before buying data rather than after. With 84 basis points of daily idiosyncratic noise, a quarterly signal on 30 names over five years — 600 events, a *generous* alternative-data sample — resolves about 7 basis points. If the hypothesized effect is smaller than that, the dataset cannot answer the question at any price, and the correct action is to decline before the trial period rather than after.

## Assume the history is backfilled until proven otherwise

Vendors sell histories. Most alternative datasets began collecting when the company was founded, so any history before that date was *reconstructed*, and reconstruction is where hindsight enters — through revised universes, through methodology tuned until the backtest looked good, through source data that itself was restated. The tell is statistical and it is easy to look for: signal quality that is conspicuously better before the vendor's go-live date than after.

```python
import numpy as np

rng = np.random.default_rng(1)
n_periods, go_live = 120, 60

for label, ic_before, ic_after in [("honest vendor", 0.03, 0.03), ("backfilled", 0.08, 0.01)]:
    z = rng.standard_normal(n_periods)
    ics = np.where(np.arange(n_periods) < go_live, ic_before, ic_after) + 0.10 * z
    pre, post = ics[:go_live], ics[go_live:]
    t = (pre.mean() - post.mean()) / np.sqrt(pre.var(ddof=1) / len(pre)
                                             + post.var(ddof=1) / len(post))
    print(f"  {label:14s}: IC before go-live {pre.mean():+.3f}, after {post.mean():+.3f}, "
          f"difference t = {t:+.2f}")
# =>   honest vendor : IC before go-live +0.025, after +0.025, difference t = +0.04
#      backfilled    : IC before go-live +0.068, after -0.007, difference t = +4.16
```

The test costs one line and separates the cases decisively: t = +0.04 for the honest history, **t = +4.16** for the backfilled one. Three practical notes make it usable. Ask the vendor for the go-live date in writing — a vendor who cannot say when live collection began has answered the question. Run the test on point-in-time universe membership too, since a "history" of the S&P 500 that uses today's membership is survivorship bias wearing a timestamp. And treat a failure as disqualifying rather than as something to adjust for: a contaminated history cannot be repaired, because the contamination is *unobserved hindsight* rather than a measurable bias, which is precisely why [Part II insisted](../part-02-python/05-sql-and-data-storage.md) that knowledge time be stored alongside event time from the first row.

## A one-day timestamp error is worth five Sharpe points

Everything above assumes you know when each observation became knowable. Alternative data makes that harder than price data, because there are usually three different timestamps — when the event happened, when the vendor observed it, and when it was delivered to you — and only the last one is honest. Satellite imagery is captured on Monday, processed Tuesday, delivered Wednesday. A filing is submitted at 4:05pm and disseminated at 4:12pm. A card panel covers a week that closed nine days ago. Using the *event* timestamp when you had the *delivery* timestamp is the single most common way an alternative-data backtest is destroyed, and the damage is easy to price:

```python
import numpy as np
import pandas as pd

px = pd.read_parquet("data/prices.parquet")
asset = np.log(px["TLT"]).diff().dropna()
signal = np.sign(asset.rolling(5).mean())            # a signal built from the last 5 days

for lag, label in [(0, "same-bar (event timestamp)"), (1, "next-bar (delivery timestamp)")]:
    s = (signal.shift(lag) * asset).dropna()
    print(f"  {label:30s}: Sharpe {np.sqrt(252) * s.mean() / s.std():+.2f}")
# =>   same-bar (event timestamp)    : Sharpe +5.27
#      next-bar (delivery timestamp) : Sharpe -0.19
```

The same signal, the same data, the same instrument: **+5.27 against −0.19**, a gap of 5.46 Sharpe points created entirely by acting on information one bar before it was available. That is a larger effect than any genuine alternative-data edge in the published literature, which is why timestamp discipline dominates modeling skill in this field. The defensive habit is mechanical: store an explicit `knowledge_time` column on every row, join with an as-of merge that respects it (`pandas.merge_asof` on knowledge time, never on event time), and set the field at *ingestion* rather than at analysis, when the delivery time is still known. A dataset whose knowledge times were reconstructed later is a dataset whose knowledge times are guesses.

## Decay, crowding, and the fifty other subscribers

Even a clean, well-timed, genuinely predictive dataset has a shelf life, because the vendor's business model is selling it to as many funds as possible. Two forces compound. **Decay**: as more capital trades the signal, the anomaly is arbitraged and the information coefficient falls, empirically at a rate well modeled as $IC_t = IC_0 e^{-t/\tau_d}$ with $\tau_d$ measured in quarters for popular datasets. **Crowding**: the remaining edge becomes correlated across subscribers, so the drawdowns arrive simultaneously and the position is hardest to exit exactly when everyone else is exiting.

The relevant arithmetic is the fundamental law of active management, $IR = IC\sqrt{BR}$, which [Part IV used](../part-04-strategy-development/03-cross-sectional-and-volatility-strategies.md) to explain why an IC of 0.012 across 108 sector bets produced an information ratio of only 0.13. Applied here, the law says something specific about alternative data: because breadth enters as a square root, a dataset covering 120 names cannot rescue a decayed IC. Halving the IC requires *quadrupling* the coverage to stand still, and coverage is exactly what alternative datasets lack — they are typically deep on a few hundred large names and absent everywhere else.

## The spreadsheet that says no

The purchase decision is arithmetic, and it should be done in the meeting. Gross alpha is $IR \times \sigma_{\text{target}} \times \text{AUM}$; costs are the subscription plus the people needed to work the data, which is where most estimates go wrong — one and a half researchers cost more than most datasets do. A dataset should clear roughly three times its fully-loaded cost before it is worth the operational risk:

```python
import numpy as np

DATA, PEOPLE = 250_000, 600_000                     # subscription + 1.5 researchers
total = DATA + PEOPLE
print(f"fully loaded annual cost: ${total / 1e3:.0f}k")
for label, ic, breadth in [("vendor's pitch deck", 0.06, 500),
                           ("after decay and crowding", 0.02, 500),
                           ("after coverage limits too", 0.02, 120)]:
    ir = ic * np.sqrt(breadth)
    line = []
    for aum in [50e6, 250e6, 1e9]:
        gross = ir * 0.10 * aum                     # 10% target volatility
        line.append(f"${aum / 1e6:>5.0f}M: ${gross / 1e6:6.2f}M")
    print(f"  {label:26s} IR {ir:.2f} | " + " | ".join(line))
ir = 0.02 * np.sqrt(120)
print(f"at a realistic IR of {ir:.2f}, the program clears 3x its cost at "
      f"${3 * total / (ir * 0.10) / 1e6:.0f}M of AUM")
# => fully loaded annual cost: $850k
#      vendor's pitch deck        IR 1.34 | $   50M: $  6.71M | $  250M: $ 33.54M | $ 1000M: $134.16M
#      after decay and crowding   IR 0.45 | $   50M: $  2.24M | $  250M: $ 11.18M | $ 1000M: $ 44.72M
#      after coverage limits too  IR 0.22 | $   50M: $  1.10M | $  250M: $  5.48M | $ 1000M: $ 21.91M
#    at a realistic IR of 0.22, the program clears 3x its cost at $116M of AUM
```

The three rows are the same dataset described by three honest parties. The vendor's assumptions make it worth buying at any size. Applying realistic decay and crowding cuts the information ratio by two-thirds, and acknowledging that the data covers 120 names rather than 500 cuts it again — to 0.22, at which point the program needs **$116M of assets** to clear three times its cost and is plainly negative at $50M.

The structural conclusion is the one worth carrying: alternative data is an **economies-of-scale business**. Costs are fixed, alpha scales with assets, so the identical dataset that is obviously worth buying at a billion dollars is obviously not at fifty million — and the vendor, whose pitch is calibrated to their largest client, will not volunteer this. For a small fund the correct alternative-data strategy is usually the one this module opened with: free, regulator-timestamped, un-backfillable sources like EDGAR, where the only cost is the research effort and the point-in-time integrity is guaranteed by the filing date.

!!! warning "The timestamp is worth more than the model"
    A one-bar timestamp error turned a Sharpe of −0.19 into +5.27 on this course's own data — larger than any genuine alternative-data edge in the literature, manufactured entirely by acting on information before it existed. Meanwhile a twelve-event study "significant" at t = 2.00 had a confidence interval spanning 0.6 to 63.5 basis points, and a backfilled vendor history announced itself at t = 4.16 to anyone who ran a two-line test. Before modeling anything: store knowledge time at ingestion, compute the minimum detectable effect before buying, and test the history for hindsight.

!!! abstract "Key takeaways"
    - Consecutive Apple 10-K risk sections are 99.77% and 98.94% identical by term frequency — companies edit rather than rewrite, so the rare changes carry the information, and the filing date is a regulator-stamped knowledge time that cannot be backfilled.
    - General-purpose sentiment lexicons mis-score filings because "liability," "cost," and "capital" are negative in English and neutral in accounting; a score is only as good as an explicit answer to what it counts.
    - Event-study resolution is $2\hat\sigma_{AR}\sqrt{L}/\sqrt{N}$: with 84 bp of daily abnormal-return noise, 12 events resolve 48.5 bp, 60 resolve 21.7, 500 resolve 7.5, and 5,000 resolve 2.4.
    - A planted 25 bp effect was measured as 24.7 bp [17.4, 32.0] at N = 500 and 32.1 bp [0.6, 63.5] at N = 12 — the second is "significant" at t = 2.00 and useless for sizing. Detection is not measurement.
    - A backfilled history was caught by comparing information coefficients before and after the vendor's go-live date: t = +4.16 against +0.04 for an honest one. Ask for the go-live date in writing.
    - A one-bar timestamp error was worth **5.46 Sharpe points** (+5.27 versus −0.19), which is larger than any real alternative-data edge — store `knowledge_time` at ingestion and join with as-of merges on it.
    - Because $IR = IC\sqrt{BR}$, halving a decayed IC requires quadrupling coverage to stand still, and coverage is exactly what alternative datasets lack.
    - Realistic assumptions cut a vendor's IR from 1.34 to 0.22, at which a $850k fully-loaded program needs **$116M of AUM** to clear three times its cost — alternative data is an economies-of-scale business, and the pitch is calibrated to the vendor's largest client.

## Where this goes next

The features this module sources have to survive the same validation gauntlet as any other, and the multiple-testing arithmetic gets worse when a vendor ships a thousand columns: [Part IV, lesson eight](../part-04-strategy-development/08-validation-and-overfitting.md) is the machinery, and [Bayesian Optimization](01-bayesian-optimization.md) shows what happens when a smarter search is pointed at a noisy objective. The point-in-time storage discipline that makes any of this trustworthy is [Part II's SQL lesson](../part-02-python/05-sql-and-data-storage.md), and the model-side handling of many weak features is [Part VII's feature engineering](../part-07-machine-learning/01-feature-engineering-for-ml.md). If the datasets under consideration are large enough that processing them becomes the constraint rather than the statistics, [GPU Acceleration](08-gpu-acceleration-cuda.md) and [Distributed Backtesting](09-distributed-backtesting.md) cover the compute — though the arithmetic in this module's last section suggests that for most funds the binding constraint is the invoice, not the hardware.
