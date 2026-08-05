# Production ML

[Reinforcement Learning and Meta-Labeling](04-reinforcement-learning-and-meta-labeling.md) ended with something this part is actually willing to run — a meta-labeled trend book at Sharpe +0.43 — and running it changes its nature. A model in research is a function; a model in production is a *process*, and [Part VI](../part-06-live-infrastructure/index.md) spent six lessons establishing what processes owe you: heartbeats, metrics, promotion gates, audit trails, and a way to turn them off. This closing lesson extends each of those obligations to the one component Part VI could not cover, because a model fails differently from a process. A crashed scheduler is *loud*; a decayed model keeps answering politely, on time, with well-formatted probabilities that have quietly stopped meaning anything. Everything here is machinery for making that silence audible.

The apparatus is Part VI's, reused rather than reinvented: the Redis instance on database 15 with its `qt:` namespace holds hot model state, the `quant` PostgreSQL database gains one table, the Prometheus text format carries five new gauges, and the `/readyz` probe learns one new word. As in Part VI, the service transcripts below were captured live against those running services while the lesson was written; the model under management is the direction classifier this part has trained a dozen times — chosen because its decay is instructively dramatic — and every procedure applies unchanged to the meta-labeler that actually earned deployment.

## Retrain cadence is an empirical question, not a policy preference

Every production ML conversation arrives at cadence: retrain never, retrain on a schedule, or learn online from every new observation. Teams usually settle this by temperament. It is a measurable question, and the frozen matrix can stage all three contenders honestly — with one subtlety the naive version always misses: a label born today [does not resolve for up to 21 sessions](01-feature-engineering-for-ml.md), so a retrainer may only consume labels whose `t1` has passed, and an online learner gets each label *the day it resolves*, not the day it forms:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
y = (spy.y_tb > 0).astype(int)
live = spy.index[spy.index >= "2017"]

def gbm():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                              seed=42, deterministic=True, force_row_wise=True,
                              num_threads=1, verbosity=-1)

# static: trained once on labels resolved by end-2016, never touched again
base = spy[spy.t1 <= "2016-12-31"]
static = gbm().fit(base[feat], y[base.index], sample_weight=base.w)
p_static = pd.Series(static.predict_proba(spy.loc[live, feat])[:, 1], index=live)

# scheduled: refit each month-start on every label resolved by then
p_month, n_refits = {}, 0
for m, chunk in spy.loc[live].groupby(pd.Grouper(freq="MS")):
    if chunk.empty:
        continue
    resolved = spy[spy.t1 < m]
    mdl = gbm().fit(resolved[feat], y[resolved.index], sample_weight=resolved.w)
    n_refits += 1
    p_month.update(dict(zip(chunk.index, mdl.predict_proba(chunk[feat])[:, 1])))
p_month = pd.Series(p_month)

# online: one SGD pass over the base years, then learn each label the day it resolves
scaler = StandardScaler().fit(base[feat])
sgd = SGDClassifier(loss="log_loss", random_state=0)
sgd.partial_fit(scaler.transform(base[feat]), y[base.index], classes=[0, 1])
p_online, n_updates = {}, 0
by_t1 = spy.groupby("t1").groups
for d in live:
    p_online[d] = sgd.predict_proba(scaler.transform(spy.loc[[d], feat]))[0, 1]
    for i in by_t1.get(d, []):                        # labels that resolved today
        sgd.partial_fit(scaler.transform(spy.loc[[i], feat]), [y[i]])
        n_updates += 1
p_online = pd.Series(p_online)

y_live = y[live]
print(f"2017+ AUC: static (never retrained) {roc_auc_score(y_live, p_static):.3f}   "
      f"monthly retrain ({n_refits} fits) {roc_auc_score(y_live, p_month):.3f}   "
      f"online SGD ({n_updates:,} updates) {roc_auc_score(y_live, p_online):.3f}")
h1 = roc_auc_score(y_live[:"2020"], p_month[:"2020"]), roc_auc_score(y_live["2021":], p_month["2021":])
h2 = roc_auc_score(y_live[:"2020"], p_static[:"2020"]), roc_auc_score(y_live["2021":], p_static["2021":])
print(f"by half: monthly {h1[0]:.3f} -> {h1[1]:.3f}   static {h2[0]:.3f} -> {h2[1]:.3f}")
# => 2017+ AUC: static (never retrained) 0.459   monthly retrain (102 fits) 0.431   online SGD (2,124 updates) 0.471
#    by half: monthly 0.414 -> 0.450   static 0.474 -> 0.446
```

The headline is a deliberate ambush: the *never-retrained* model beat the monthly retrainer, 0.459 to 0.431, and a linear online learner beat both — one hundred and two faithful refits bought negative value. On a signal this thin the mechanism is unsurprising once named: each refit re-estimates the same near-zero structure plus a fresh draw of noise, and chasing the most recent labels means chasing the noise most of all, while the static model at least holds one noise-draw still. But the second line is why the section title says *empirical*: split the live years in half and the streams cross — the static model decays (0.474 falling to 0.446) exactly as the staleness story predicts, while the monthly retrainer *improves* (0.414 rising to 0.450) as its training window grows past the noise-dominated regime. Neither dogma survives contact: "models decay, retrain often" and "retraining chases noise" are both true, at different times, at different signal strengths — the useful output of this experiment is not a winner but a *measurement protocol*, the same three streams your own system should score continuously. And note what all three AUCs have in common: a coin, within noise. The cadence machinery matters not because it rescues this weak model but because [lesson four's meta-model](04-reinforcement-learning-and-meta-labeling.md) — the one with actual production value — will decay by the same mechanics, and by then the measurement had better already be running.

## Drift is the permanent condition, so alarm on its rank, not its presence

The standard drift toolkit has three layers, ordered by how early they can fire: monitors on *features* (the inputs left the training distribution — no labels needed, fires immediately), on *predictions* (the model's output distribution shifted — also label-free), and on *outcomes* (realized hit rate broke its control limits — definitive and latest, since labels take days to resolve). The feature layer's standard instrument is the population stability index, with thresholds inherited from credit scoring: 0.10 means investigate, 0.25 means act. Run all three on a model trained through 2006 and monitored ever after, and financial data teaches the toolkit a lesson:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
y = (spy.y_tb > 0).astype(int)

train = spy[spy.index <= "2006"]                      # the model's world ends in 2006
mdl = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                         seed=42, deterministic=True, force_row_wise=True,
                         num_threads=1, verbosity=-1
                         ).fit(train[feat], y[train.index], sample_weight=train.w)
ref = train.loc["2004":]                              # PSI reference: the regime it last saw
mon = spy[spy.index >= "2007"]
p = pd.Series(mdl.predict_proba(mon[feat])[:, 1], index=mon.index)

def psi(win, ref_vals):
    edges = np.quantile(ref_vals, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    rf = np.histogram(ref_vals, bins=edges)[0] / len(ref_vals)
    fr = np.histogram(win, bins=edges)[0] / len(win)
    fr = np.where(fr == 0, 1e-4, fr)
    return ((fr - rf) * np.log(fr / rf)).sum()

frozen = ref.f_vol_21.values
psi_m = mon.f_vol_21.rolling(252).apply(lambda w: psi(w, frozen), raw=True
                                        ).resample("MS").last().dropna()
print(f"PSI(f_vol_21, trailing year vs the frozen 2004-06 reference) > 0.25: "
      f"{(psi_m > 0.25).sum()} of {len(psi_m)} months")

yearly = {yr: psi(spy.loc[str(yr), "f_vol_21"], spy.loc[str(yr - 3):str(yr - 1), "f_vol_21"])
          for yr in range(2007, 2025)}                # rolling 3y reference, retrained annually
top = sorted(yearly, key=yearly.get, reverse=True)
print(f"vs a rolling 3-year reference, every year of 18 still exceeds 0.25 - "
      f"the ranking carries the signal:")
print(f"  loudest: " + ", ".join(f"{y} ({yearly[y]:.1f})" for y in top[:4]) +
      f";  quietest: {top[-1]} ({yearly[top[-1]]:.2f})")

mu0 = p.loc[:"2007-12"].mean()                        # CUSUM on the prediction stream
s_pos, s_neg, k, fired = 0.0, 0.0, 0.01, None
for d, x in p.loc["2008":].items():
    s_pos, s_neg = max(0, s_pos + x - mu0 - k), max(0, s_neg + mu0 - x - k)
    if max(s_pos, s_neg) > 2.0 and fired is None:
        fired = d
print(f"CUSUM on mean prediction (mu0 {mu0:.3f} from 2007): first alarm {fired:%Y-%m-%d}")

hit = (np.sign(p - 0.5) == np.sign(y[mon.index] - 0.5)).rolling(63).mean()
band = 2 * np.sqrt(0.5 * 0.5 / 63)
low = hit[hit < 0.5 - band]
print(f"hit-rate 63d below binomial control limit ({0.5 - band:.3f}): "
      f"{len(low)} of {hit.notna().sum():,} days, worst {hit.min():.3f} on {hit.idxmin():%Y-%m-%d}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(list(yearly), list(yearly.values()))
ax.axhline(0.25, ls="--", color="tab:red", label="textbook threshold 0.25")
ax.set_ylabel("PSI vs rolling 3-year reference")
ax.legend()
plt.show()
# => PSI(f_vol_21, trailing year vs the frozen 2004-06 reference) > 0.25: 193 of 210 months
#    vs a rolling 3-year reference, every year of 18 still exceeds 0.25 - the ranking carries the signal:
#      loudest: 2008 (5.8), 2017 (5.4), 2022 (4.0), 2023 (3.4);  quietest: 2019 (0.50)
#    CUSUM on mean prediction (mu0 0.605 from 2007): first alarm 2008-01-10
#    hit-rate 63d below binomial control limit (0.374): 907 of 4,585 days, worst 0.063 on 2018-01-23
```

The first line detonates the imported threshold: against the frozen training reference, PSI exceeds "act now" in 193 of 210 months — the regime the model was trained in *effectively never returned*, which is simultaneously the strongest argument for retraining this course can print and, as an alarm, useless, because [Part VI's alerting lesson](../part-06-live-infrastructure/04-monitoring-logging-alerting.md) already established what a siren that never stops teaches its operators. The second line removes the excuse: even against a rolling three-year reference — the fairest one a monthly retrainer could claim — every single year of eighteen "acts." Credit-scoring thresholds assume populations that hold still between quarterly reviews; financial distributions do not hold still, ever, so drift here is not an event but the *permanent condition*, and the operational move is to alarm on the *rank* of the statistic against its own history rather than its presence above an imported constant. Read the bar chart with that eye and it works beautifully: the loudest years are 2008 (5.8), 2017 (5.4), and 2022 (4.0) — the crisis, the great vol *collapse*, and the bear market — with 2017 the teaching moment, because drift monitors are two-sided and a world going quiet is as much not-your-training-world as a world on fire. The last two lines calibrate the layers' latencies: the prediction-stream CUSUM, needing no labels at all, threw its first alarm on **2008-01-10** — nine months before Lehman — while the outcome monitor is definitive but chronic (907 days below its control limit; a 63-session stretch in January 2018 where the stale model was right 6% of the time, an anti-model). Early layers whisper early; late layers shout late; the [regime-detection appendix](../appendix/part-18-quant-finance-applications/15-regime-detection.md) supplies the formal machinery under all three.

## Model health is a metrics endpoint

Part VI's rule was that a monitoring number is only trustworthy if derived from the system of record by an independent path, and its transport was deliberately humble: `name{labels} value`, one per line, the Prometheus exposition format. Model health rides the same rails — five gauges, computed from the frozen matrix and the champion's own predictions, with the champion's identity in Redis where every other `qt:` mark lives:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import redis
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
y = (spy.y_tb > 0).astype(int)
base = spy[spy.t1 <= "2016-12-31"]
mdl = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                         seed=42, deterministic=True, force_row_wise=True,
                         num_threads=1, verbosity=-1
                         ).fit(base[feat], y[base.index], sample_weight=base.w)
tail = spy.iloc[-63:]                                 # the trailing quarter, as live would see it
p = mdl.predict_proba(tail[feat])[:, 1]

edges = base.f_vol_21.quantile(np.linspace(0, 1, 11)).values.copy()
edges[0], edges[-1] = -np.inf, np.inf
ref = np.histogram(base.f_vol_21, bins=edges)[0] / len(base)
frac = np.histogram(tail.f_vol_21, bins=edges)[0] / len(tail)
frac = np.where(frac == 0, 1e-4, frac)
psi_now = ((frac - ref) * np.log(frac / ref)).sum()

r = redis.Redis(db=15, decode_responses=True)          # the hot store of Part VI, db 15
r.set("qt:model:champion", "dir-lgbm-2016", ex=86400)
metrics = [
    ("qt_model_auc_63d", f"{roc_auc_score(y[tail.index], p):.3f}"),
    ('qt_model_psi{feature="f_vol_21"}', f"{psi_now:.3f}"),
    ("qt_model_pred_mean_63d", f"{p.mean():.3f}"),
    ("qt_model_hit_rate_63d", f"{(np.sign(p - .5) == np.sign(y[tail.index] - .5)).mean():.3f}"),
    ("qt_model_age_days", (pd.Timestamp("2025-06-23") - base.t1.max()).days),
]
for name, val in metrics:                              # the exposition format of Part VI, lesson four
    print(name, val)

POLICY = {                                             # severity contracts, same three verbs
    "qt_model_psi > 0.25 for 5 sessions": "TICKET — schedule retrain, widen validation",
    "qt_model_auc_63d < 0.45":            "TICKET — challenger evaluation moves up",
    "qt_model_age_days > retrain SLA":    "PAGE only if /readyz flips; else TICKET",
    "qt_model_pred_mean shift (CUSUM)":   "LOG until confirmed by a second monitor",
}
print(f"champion in redis: {r.get('qt:model:champion')}")
for cond, act in POLICY.items():
    print(f"  {cond:38s} -> {act}")
# => qt_model_auc_63d 0.080
#    qt_model_psi{feature="f_vol_21"} 2.758
#    qt_model_pred_mean_63d 0.575
#    qt_model_hit_rate_63d 0.508
#    qt_model_age_days 3097
#    champion in redis: dir-lgbm-2016
#      qt_model_psi > 0.25 for 5 sessions     -> TICKET — schedule retrain, widen validation
#      qt_model_auc_63d < 0.45                -> TICKET — challenger evaluation moves up
#      qt_model_age_days > retrain SLA        -> PAGE only if /readyz flips; else TICKET
#      qt_model_pred_mean shift (CUSUM)       -> LOG until confirmed by a second monitor
```

The gauges describe a model eight and a half years past its training data — `qt_model_age_days` 3097 — and every needle confirms it: PSI at 2.758 against its own training reference, a trailing-quarter AUC of 0.080. That last number needs the caution it teaches: sixty-three sessions of AUC carries enormous sampling variance (the same model's hit rate in the same window is an unremarkable 0.508), so a single wild reading is a *prompt*, not a verdict — which is exactly why the policy table demands persistence ("for 5 sessions") and routes single-monitor signals to `TICKET` rather than `PAGE`. The severity mapping inherits Part VI's sharp test — *is there an action only a human can take, and does it matter tonight?* — and model decay almost never passes it: no model emergency at 2 a.m. is improved by a groggy human retraining anything, so the honest severities are `TICKET` (a human decides tomorrow, with coffee), `LOG` (one monitor muttering, awaiting confirmation), and `PAGE` reserved for the single case where model health blocks *trading* — the readiness flip that section six wires up. A dashboard drawn from these five gauges answers, at a glance, the question every morning check should open with: *is the thing making our predictions still the thing we measured?*

## Champion, challenger, and the gate between them

The monitors having voted for a retrain, the dangerous move is the obvious one: train the new model and swap it in, on the builder's confidence. [Part VI's promotion gate](../part-06-live-infrastructure/06-secrets-paper-live-compliance.md) exists because the author's confidence is the least reliable instrument in the building, and its form transfers to models unchanged: the incumbent (*champion*) keeps trading; the candidate (*challenger*) runs in shadow, scored on data neither has trained on; criteria are written *before* the run; the verdict names its blocker. Two challengers face the gate below, and the shadow window — 2022 through 2023, a bear and a recovery — is deliberately after every training cutoff:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
y = (spy.y_tb > 0).astype(int)
shadow = spy[(spy.index >= "2022") & (spy.index <= "2023")]

def train_through(cutoff):
    d = spy[spy.t1 <= cutoff]
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                              seed=42, deterministic=True, force_row_wise=True,
                              num_threads=1, verbosity=-1
                              ).fit(d[feat], y[d.index], sample_weight=d.w)

def shadow_stats(m):
    p = m.predict_proba(shadow[feat])[:, 1]
    return roc_auc_score(y[shadow.index], p), brier_score_loss(y[shadow.index], p)

champ = train_through("2018-12-31")
a0, b0 = shadow_stats(champ)
print(f"champion (through 2018): shadow AUC {a0:.3f}, Brier {b0:.4f}, "
      f"n = {len(shadow)} sessions")

GATE = [("shadow sessions >= 250",  lambda a, b, n: n >= 250),   # written before the run
        ("AUC beats champion by 0.010", lambda a, b, n: a >= a0 + 0.010),
        ("Brier no worse than champion", lambda a, b, n: b <= b0)]

for name, cutoff in [("challenger-2021", "2021-12-31"), ("challenger-2020H1", "2020-06-30")]:
    a, b = shadow_stats(train_through(cutoff))
    failed = [g for g, fn in GATE if not fn(a, b, len(shadow))]
    verdict = "PROMOTE" if not failed else f"HOLD — fails {failed[0]!r}"
    print(f"{name} (through {cutoff[:7]}): AUC {a:.3f}, Brier {b:.4f} -> {verdict}")
print("rollback is a registry status flip; champion artifacts are never deleted")
# => champion (through 2018): shadow AUC 0.530, Brier 0.2609, n = 251 sessions
#    challenger-2021 (through 2021-12): AUC 0.470, Brier 0.3500 -> HOLD — fails 'AUC beats champion by 0.010'
#    challenger-2020H1 (through 2020-06): AUC 0.581, Brier 0.2660 -> HOLD — fails 'Brier no worse than champion'
#    rollback is a registry status flip; champion artifacts are never deleted
```

Both challengers lost, differently, and each HOLD is the gate doing a distinct job. The 2021 challenger is the straightforward case: three more years of training data produced a *worse* shadow AUC (0.470 against the champion's 0.530) — section one's noise-chasing mechanism, caught at the gate instead of in production. The 2020H1 challenger is the reason gates have more than one criterion: its AUC of 0.581 *beats* the champion by five points — a naive gate promotes it on the spot — but its Brier of 0.2660 is worse, meaning its probabilities are more confident than its accuracy justifies (a model that drank 2020's chaos and came out swaggering), and [lesson two](02-tree-ensembles.md) established what happens downstream when position sizing consumes overconfident probabilities. A ranking metric and a calibration metric disagree; the gate's ordering says calibration is load-bearing; HOLD. Note the two properties inherited from Part VI verbatim: the criteria predate the run — written after, they would be fitted to whichever challenger someone already favored, [Part IV's sin](../part-04-strategy-development/08-validation-and-overfitting.md) in operational costume — and the verdict names its blocker, so each HOLD arrives with a work order. The final line is the lesson's title warning in miniature: promotion is an `UPDATE ... SET status`, demotion is the same statement backwards, and the artifact of every model that ever held the champion slot stays on disk forever — because the fastest recovery from a bad promotion is the one that requires no retraining at all.

## A registry, or a model you cannot answer for

Part VI closed on audit chains: a trade you cannot reconstruct carries no evidentiary weight. A live prediction has the same standard and one more dependency — it was made by a *model*, and "which model?" must have an exact answer years later, when the person who trained it is gone and the training script has been refactored twice. The registry is that answer as a table, and its two hash columns are the load-bearing ones: the artifact hash fingerprints the trained model itself (LightGBM's `model_to_string` serializes the full tree ensemble as stable text), and the dataset hash fingerprints `data/part7.parquet` byte-for-byte — [lesson one's frozen file](01-feature-engineering-for-ml.md) earning its freeze a fourth time:

```python
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import psycopg
from sklearn.metrics import roc_auc_score

spy = pd.read_parquet("data/part7.parquet").query("symbol == 'SPY'")
feat = [k for k in spy.columns if k.startswith("f_")]
y = (spy.y_tb > 0).astype(int)
shadow = spy[(spy.index >= "2022") & (spy.index <= "2023")]

def build(model_id, cutoff, status):
    d = spy[spy.t1 <= cutoff]
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                           seed=42, deterministic=True, force_row_wise=True,
                           num_threads=1, verbosity=-1
                           ).fit(d[feat], y[d.index], sample_weight=d.w)
    art = hashlib.sha256(m.booster_.model_to_string().encode()).hexdigest()[:12]
    auc = roc_auc_score(y[shadow.index], m.predict_proba(shadow[feat])[:, 1])
    return (model_id, cutoff, art, auc, status)

data_sha = hashlib.sha256(open("data/part7.parquet", "rb").read()).hexdigest()[:12]
rows = [build("dir-lgbm-2018", "2018-12-31", "champion"),
        build("dir-lgbm-2021", "2021-12-31", "shadow")]

with psycopg.connect("dbname=quant") as conn:          # the durable store of Part VI
    conn.execute("""CREATE TABLE IF NOT EXISTS models (
        model_id text PRIMARY KEY, trained_through date, artifact_sha text,
        dataset_sha text, feature_list text, shadow_auc real, status text,
        created_at timestamptz DEFAULT now())""")
    for mid, cut, art, auc, status in rows:
        conn.execute("""INSERT INTO models
            (model_id, trained_through, artifact_sha, dataset_sha, feature_list,
             shadow_auc, status) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO UPDATE SET artifact_sha = EXCLUDED.artifact_sha,
                shadow_auc = EXCLUDED.shadow_auc, status = EXCLUDED.status,
                created_at = now()""", (mid, cut, art, data_sha, ",".join(feat), auc, status))
    for row in conn.execute("""SELECT model_id, trained_through, artifact_sha,
                               dataset_sha, shadow_auc, status FROM models
                               ORDER BY trained_through"""):
        print(row)
    mid, art, dat = conn.execute("""SELECT model_id, artifact_sha, dataset_sha
                                    FROM models WHERE status = 'champion'""").fetchone()
print(f"lineage of any live prediction: model {mid}, artifact sha256:{art}, "
      f"trained on dataset sha256:{dat} - rebuildable from data/part7.parquet and a seed")
# => ('dir-lgbm-2018', datetime.date(2018, 12, 31), '38c8b5a0a5af', 'b956966146cd', 0.52988404, 'champion')
#    ('dir-lgbm-2021', datetime.date(2021, 12, 31), '82aaf0a10947', 'b956966146cd', 0.4703071, 'shadow')
#    lineage of any live prediction: model dir-lgbm-2018, artifact sha256:38c8b5a0a5af, trained on dataset sha256:b956966146cd - rebuildable from data/part7.parquet and a seed
```

Two rows, and the audit chain closes. Both models share the dataset hash `b956966146cd` — they were trained on *the same bytes*, provably, which is the claim no amount of "we used the July data" documentation can make — while their artifact hashes differ because the artifacts do: change one tree in one booster and `38c8b5a0a5af` becomes something else, silently and detectably. This is why the block hashes the *serialized model* rather than a pickle (pickles embed library versions and memory layout; the same model can pickle to different bytes) and why `feature_list` rides along as its own column: the classic production incident is not a bad model but a good model fed columns in the wrong order, and the registry row is where the feed's schema gets checked against the model's expectation. The deliberate smallness of the table is its virtue. MLflow, SageMaker, and their peers wrap this same core in UI and lifecycle hooks, and adopting them later is a migration, not a redesign — but the eight columns above are the part a trading desk cannot outsource, because they answer the interview [Part VI's checklist](../part-06-live-infrastructure/06-secrets-paper-live-compliance.md) said every incident conducts: *which model made this prediction, trained on what, measured how, promoted when, by what right.* A model you cannot answer for is, to a post-mortem, indistinguishable from a model you never tested.

## The model joins the nightly DAG

One wire remains unconnected: the registry knows the champion's age, the policy table says staleness matters, but nothing yet *refuses to trade* on a stale model. Part VI's instrument for "the process is fine and must not trade" is the readiness probe — `/healthz` says alive, `/readyz` says safe — and the model becomes a first-class citizen of the trading system at the moment its freshness becomes a named check in that probe:

```python
import http.server
import json
import threading

import psycopg
import redis
import requests

r = redis.Redis(db=15, decode_responses=True)
SLA_DAYS = 35                                          # a monthly retrainer, with slack

def champion_age_days():
    with psycopg.connect("dbname=quant", connect_timeout=1) as c:
        row = c.execute("""SELECT extract(epoch FROM now() - created_at) / 86400
                           FROM models WHERE status = 'champion'""").fetchone()
        return float(row[0]) if row else 999.0

class Health(http.server.BaseHTTPRequestHandler):      # the probe of Part VI, lesson four
    def do_GET(self):
        body = {"redis": r.ping(), "postgres": True,
                "model_fresh": champion_age_days() <= SLA_DAYS}
        code = 200 if all(body.values()) else 503
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"

with psycopg.connect("dbname=quant") as conn:          # last night's retrain just stamped it
    conn.execute("UPDATE models SET created_at = now() WHERE status = 'champion'")
resp = requests.get(base + "/readyz")
print("/readyz", resp.status_code, resp.json())

with psycopg.connect("dbname=quant") as conn:          # six missed retrains later...
    conn.execute("""UPDATE models SET created_at = now() - interval '40 days'
                    WHERE status = 'champion'""")
resp = requests.get(base + "/readyz")
print("/readyz", resp.status_code, resp.json())
srv.shutdown()

DAG = ["fetch_bars (18:00)", "validate_bars", "features + predict  <- gated on model_fresh",
       "risk_checks", "stage_orders (next open)"]
print(" -> ".join(DAG))
print("monthly, after the 1st's validate_bars: retrain -> shadow-score -> gate -> registry")
# => /readyz 200 {'redis': True, 'postgres': True, 'model_fresh': True}
#    /readyz 503 {'redis': True, 'postgres': True, 'model_fresh': False}
#    fetch_bars (18:00) -> validate_bars -> features + predict  <- gated on model_fresh -> risk_checks -> stage_orders (next open)
#    monthly, after the 1st's validate_bars: retrain -> shadow-score -> gate -> registry
```

The transcript is Part VI's mark-expiry scene with a new actor: the registry timestamp goes stale, `/readyz` flips 200 to 503 *with the failing check named in the body*, and everything downstream inherits the refusal — the prediction job does not run, so orders are not staged, so a model past its service interval cannot quietly keep trading, no vigilance required. The failure this collapses is insidious in the unmonitored version: the retrain job dies in March, nobody notices because predictions keep flowing (the old champion answers happily forever), and the system is discovered running a nine-month-old model in November by someone investigating unrelated losses — section three's gauges *describe* that scene; the probe *prevents* it. The DAG line places the model in [Part VI's job ordering](../part-06-live-infrastructure/02-scheduling-and-data-plumbing.md) with prediction downstream of validated bars and gated on freshness, while the retrain pipeline runs as its own monthly branch whose terminal node is not "deploy" but *"registry"* — a new row, status `shadow`, awaiting the gate. Deployment, in a system built this way, is never an action someone performs; it is a status a model earns.

!!! warning "A model you cannot roll back is a position you cannot close"
    Every mechanism in this lesson converges on reversibility. The registry keeps every artifact that ever held the champion slot; promotion and rollback are the same one-row status flip in opposite directions; the gate exists so the flip forward is earned, and the archive exists so the flip backward is instant. This is risk management applied to the meta-level: a position without an exit is a risk you cannot bound, and a deployed model without a rollback is exactly that — when it misbehaves (and section two established that drift is the permanent condition, so it will), your recovery time is either one `UPDATE` statement or one emergency retrain performed at the worst possible moment by the most stressed possible person. Blue-green deploys, feature flags, and champion/challenger are one idea wearing three costumes: the old version stays runnable until long after the new one has proven itself. Never delete a model the day you demote it; the day you demote it is precisely when you have least information about whether the demotion was right.

!!! abstract "Key takeaways"
    - Retrain cadence is measurable, not doctrinal: the never-retrained model beat 102 monthly refits on aggregate (AUC 0.459 vs 0.431) while the halves crossed — static decayed 0.474 → 0.446 as monthly climbed 0.414 → 0.450 — so run all three streams and let the measurement pick.
    - Against its frozen training reference, PSI exceeded the "act" threshold in 193 of 210 months, and every year of 18 exceeded it against a rolling reference: on financial features drift is the permanent condition — alarm on the statistic's rank (2008: 5.8, 2017: 5.4, 2022: 4.0), not its presence.
    - The label-free CUSUM on the prediction stream fired 2008-01-10, nine months before the outcome monitors could know; the layers' latencies — features, predictions, outcomes — are the design.
    - Model health became five `qt_model_*` gauges in Part VI's exposition format, with severities that pass the only-a-human test: decay is a `TICKET` with coffee, never a 2 a.m. `PAGE` — except when it blocks trading, which is the probe's job.
    - The promotion gate HELD both challengers for different reasons — one worse outright (AUC 0.470), one *better* on AUC (0.581) but overconfident on Brier (0.2660 vs 0.2609) — criteria written before the run, verdicts naming their blockers.
    - The registry's two hashes close the audit chain: both models provably trained on dataset `b956966146cd`, each artifact fingerprinted (`38c8b5a0a5af` vs `82aaf0a10947`), every live prediction traceable to exact model, data, and features.
    - `/readyz` learned `model_fresh`: a champion older than its retrain SLA flips 200 → 503 and the prediction job simply does not run — staleness became an enforced contract instead of a hoped-for habit.

## Where this goes next

This closes Part VII, and the part's honest ledger deserves one restatement: direction prediction failed at every capacity tried — trees, boosters, four deep architectures, a reinforcement learner — while the disciplined periphery paid: a leak-audited dataset, purged validation that refused false discoveries, meta-labeling that lifted a real strategy from +0.29 to +0.43, an ensemble that halved the drawdown of its best member, and now the operational apparatus that lets any of it run unattended without rotting in secret. That is what "applying ML responsibly" turned out to mean: the machine learning earned its keep everywhere *except* the place the brochures promised. But everything this part built manages *one* signal on *one* book at a time, and a desk is neither: it is many signals, many assets, and one pool of capital whose risk must be measured, budgeted, and survived. [Part VIII — Portfolio Management](../part-08-portfolio-management/index.md) takes over there, beginning where every allocation decision must: with measuring the risk you already have.
