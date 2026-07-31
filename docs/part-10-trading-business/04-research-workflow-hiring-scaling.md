# Research Workflow, Hiring, and Scaling

[Operations, Compliance, and Tax](03-operations-compliance-tax.md) built a calendar that keeps an existing business running. It does not produce the next strategy, and a book that stops adding sleeves has a known trajectory: the edge decays, capacity fills, and the operations run flawlessly on a strategy that stopped working two years ago. This lesson is about the process that produces strategies rather than trades — and about the people and structure that outlast any one of them.

The measurement is uncomfortable in a specific way. Running a thousand ideas through the screen most desks actually use, the survivors **promise a Sharpe of 1.25 and deliver 0.19 — an 85% haircut** — and only a third of them have any edge at all. The statistically correct correction for having tested a thousand ideas rejects **every single one**. What closes the gap is neither the lenient gate nor the correct one, and the second block prices the answer: sustaining a six-sleeve book needs **333 ideas a year** and a research organization that costs **$38 million of AUM** to fund.

!!! note "Scope"
    Both blocks are self-contained and seeded. The first uses a deliberately generous world — one idea in ten is real, with a true Sharpe of 0.50 — so that the failure it measures is a lower bound on the real one. [Part IV](../part-04-strategy-development/08-validation-and-overfitting.md) established the multiple-testing problem for one researcher tuning one strategy; this lesson applies it to the firm.

## A pipeline is a set of places to say no

A research pipeline is not a workflow diagram, it is a sequence of gates, and a gate is only real if it has a written evidence standard and a record of things it rejected. The stages differ between firms; the property that matters is that each one costs more than the last and each one is allowed to kill.

| Stage | Evidence required to advance | Cost of the stage |
|---|---|---|
| Intake | A written hypothesis naming the economic reason the edge exists and who is on the other side | An hour |
| Screen | Signal survives a crude test on held-out data with realistic costs applied | A day |
| Backtest | Full engine run, cost and capacity modelled, sensitivity to parameters flat, no lookahead in review | A week or two |
| Paper | Live signal generation against live data, no capital, decisions logged and compared to the backtest | A month, plus infrastructure |
| Small live | Real capital at a size where the loss is affordable; slippage and fills match the model | A quarter, plus real money |
| Allocation | Correlation with the existing book is low enough to earn its risk budget | A committee meeting |

The last gate is the one most often missing, and its absence is why so many books contain three versions of the same trade. A strategy that passes every standalone test and correlates 0.8 with an existing sleeve adds turnover, capacity pressure, and operational surface without adding diversification — the [Part VIII](../part-08-portfolio-management/03-risk-parity-diversification-factors.md) result that the marginal contribution of a sleeve depends on the book it joins, not on its own statistics.

The intake gate is the cheapest and the most under-used. Requiring one paragraph naming the counterparty — who is losing the money, and why they keep doing it — kills a large fraction of ideas before anyone writes code, and it costs an hour.

## The survivors promise what they cannot deliver

Here is what a screen does to a thousand ideas in a world deliberately kinder than the real one: one in ten has genuine edge, and edge means a true Sharpe of 0.50.

```python
import numpy as np

rng = np.random.default_rng(7)
N, OOS, VOL = 1000, 504, 0.01               # ideas tested, out-of-sample days, daily vol

# One idea in ten is real, and "real" here is generous: a true Sharpe of 0.5.
real = rng.random(N) < 0.10
mu = np.where(real, 0.50, 0.00) / np.sqrt(252) * VOL


def sharpe(r):
    return r.mean(axis=1) / r.std(axis=1, ddof=1) * np.sqrt(252)


def trial(days, cut, label):
    ins = sharpe(rng.normal(mu[:, None], VOL, (N, days)))
    oos = sharpe(rng.normal(mu[:, None], VOL, (N, OOS)))
    keep = ins > cut
    k = int(keep.sum())
    if not k:
        print(f"  {label:28s} {days // 252:3d}y {cut:5.2f} {k:6d}   nothing survives")
        return
    i_bar, o_bar = ins[keep].mean(), oos[keep].mean()
    print(f"  {label:28s} {days // 252:3d}y {cut:5.2f} {k:6d} "
          f"{real[keep].mean():6.0%} {i_bar:9.2f} {o_bar:9.2f} {1 - o_bar / i_bar:8.0%}")


print(f"  {N} ideas tested, {real.mean():.0%} of them real at a true Sharpe of 0.50")
print(f"  {'gate':28s} {'IS':>4s} {'cut':>5s} {'kept':>6s} {'real':>6s} "
      f"{'promised':>9s} {'delivered':>9s} {'haircut':>8s}")
trial(756, 1.00, "the screen everyone uses")
trial(756, 1.50, "a stricter screen")
trial(756, np.sqrt(2 * np.log(N)) * np.sqrt(252 / 756), "corrected for 1000 trials")
trial(2520, 1.00, "the same screen, more data")
print("  the correction needs N, and N is the number a registry exists to record")
# =>   1000 ideas tested, 10% of them real at a true Sharpe of 0.50
#      gate                           IS   cut   kept   real  promised delivered  haircut
#      the screen everyone uses       3y  1.00     53    32%      1.25      0.19      85%
#      a stricter screen              3y  1.50     12    50%      1.67      0.55      67%
#      corrected for 1000 trials      3y  2.15      0   nothing survives
#      the same screen, more data    10y  1.00      9   100%      1.11      0.55      50%
#      the correction needs N, and N is the number a registry exists to record
```

The first row is the industry standard and it is a disaster. A three-year backtest and a Sharpe-above-1.0 screen keeps **53 ideas of 1000, of which only 32% have any edge**. Those 53 promise an average Sharpe of **1.25** and deliver **0.19** out of sample — an **85% haircut**, and 0.19 is well below what it costs to run a sleeve. Every one of those 53 has a backtest a researcher would present with a straight face.

Tightening the same screen helps and does not fix it. At a 1.50 cut, **12 survive, half of them real, promising 1.67 and delivering 0.55** — still a 67% haircut, and the process just discarded a lot of genuine ideas to get there. The third row is the honest correction: with 1000 trials on three years of data, the threshold that controls the false-discovery rate is a Sharpe of **2.15**, and **nothing survives it**. That is not a bug in the correction. Three years of daily data cannot distinguish a true Sharpe of 0.5 from noise when you have looked a thousand times, and no threshold can conjure the missing information.

The fourth row is the answer, and it is not a threshold at all. **The same lenient Sharpe-1.0 screen, applied to ten years of data instead of three, keeps 9 ideas — 100% of them real** — promising 1.11 and delivering 0.55. Compare it to row two: the same 0.55 out of sample, but reached with nine clean ideas rather than twelve of which half are noise. **Selection bias is fixed with more data per idea and fewer ideas, not with a stricter gate on the same data.** That is a resourcing decision and a discipline decision, which is why it is in this lesson rather than a statistics one.

!!! warning "The correction needs a number you do not have"
    Every threshold in rows two and three depends on N — how many ideas were tested. Not how many were written up, or presented, or remembered: how many were *tested*, including the eleven variants tried on a Tuesday and abandoned, and the parameter sweep that counts as hundreds. A firm without a registry cannot compute its own N, so it cannot apply any correction, so it defaults to row one. This is the entire practical argument for the registry, and it is stronger than the usual one about institutional memory.

## The registry is the only thing that knows N

A research registry is a table with one row per tested idea, written at intake and closed at kill or allocation. It is boring and it is the highest-leverage artefact in a research organization.

| Field | Why it is there |
|---|---|
| `id`, `date_opened`, `researcher` | Attribution and a countable N |
| `hypothesis` | The economic reason, in a sentence, written before the test |
| `kill_criteria` | What result would end this, declared before the result exists |
| `data`, `universe`, `period` | What was tested, so a repeat is recognized as a repeat |
| `variants_tested` | The real contribution to N — a sweep is not one trial |
| `stage_reached`, `date_closed` | Where it died |
| `kill_reason` | Free text, and the field people actually read later |
| `commit` | The [Part IX](../part-09-software-engineering/01-git-and-code-review.md) manifest hash, so the result is reproducible |

Two fields do most of the work. `kill_criteria`, written before the test, is what converts a disappointing result from a negotiation into an outcome — the difference between "the Sharpe was 0.6, let me try a different lookback" and "we said below 0.8 kills it". And `variants_tested` is the one that makes N honest, because the gap between a firm's remembered idea count and its actual trial count is usually an order of magnitude, and the correction moves with the logarithm of it.

The graveyard is the point, not a side effect. A registry that records only successes is a track record; a registry that records the 991 failures is a research asset, because it stops the same idea being retested every eighteen months by each new hire, and because the `kill_reason` column is where the firm's actual knowledge accumulates. Quarterly, read the last quarter's kills: ideas killed for the same reason repeatedly are pointing at a data problem or a missing tool, not at a run of bad luck.

Post-mortems belong to the live book too, and on a schedule rather than on trauma. A sleeve that is retired gets the same write-up as one that is killed in research: what it was supposed to do, what it did, whether the failure was the thesis, the implementation, or the regime, and what would have caught it sooner.

## What a six-sleeve book actually costs

The pass rates above compose. Given them, plus how long a sleeve survives before it decays, the size of the research organization is arithmetic rather than ambition.

```python
import numpy as np

# Pass rates through the pipeline, and the capital a stage is allowed to risk.
gates = [
    ("intake to screen", 0.20, 0),
    ("screen to backtest", 0.25, 0),
    ("backtest to paper", 0.30, 0),
    ("paper to small live", 0.50, 25_000),
    ("small live to allocation", 0.60, 250_000),
]
LIFE, PER_RESEARCHER, COST = 4.0, 60, 220_000     # sleeve years, ideas/researcher-year, loaded
FEE_RATE = 0.032                                  # manager revenue per $1 AUM, from lesson one

survive = np.cumprod([g[1] for g in gates])
print(f"  {'stage':26s} {'pass':>6s} {'of 1000':>8s} {'capital at stage':>17s}")
for (name, rate, cap), s in zip(gates, survive):
    print(f"  {name:26s} {rate:6.0%} {1000 * s:8.1f} {cap:17,}")
end_to_end = float(survive[-1])
print(f"  end to end {end_to_end:.2%}: {1 / end_to_end:,.0f} ideas per allocation, "
      f"and a sleeve lasts {LIFE:.0f} years")

print(f"\n  {'target sleeves':>14s} {'ideas/yr':>9s} {'researchers':>12s} "
      f"{'research cost':>14s} {'AUM to fund it':>15s}")
for target in (1, 3, 6, 10):
    ideas = target / (end_to_end * LIFE)
    heads = ideas / PER_RESEARCHER
    spend = heads * COST
    print(f"  {target:14d} {ideas:9.0f} {heads:12.1f} {spend:14,.0f} "
          f"{spend / FEE_RATE:15,.0f}")
# =>   stage                        pass  of 1000  capital at stage
#      intake to screen              20%    200.0                 0
#      screen to backtest            25%     50.0                 0
#      backtest to paper             30%     15.0                 0
#      paper to small live           50%      7.5            25,000
#      small live to allocation      60%      4.5           250,000
#      end to end 0.45%: 222 ideas per allocation, and a sleeve lasts 4 years
#
#      target sleeves  ideas/yr  researchers  research cost  AUM to fund it
#                   1        56          0.9        203,704       6,365,741
#                   3       167          2.8        611,111      19,097,222
#                   6       333          5.6      1,222,222      38,194,444
#                  10       556          9.3      2,037,037      63,657,407
```

**222 ideas per allocation.** That is the number to internalize, and it is in the same territory as the previous block, where a properly powered screen kept 9 per 1000 before any of the process gates ran. A researcher who has tried thirty things and found nothing is not underperforming; they are one seventh of the way to a result.

The capital column is the part that is usually improvised. **Paper costs nothing, small live risks $25,000, allocation risks $250,000** — a ladder in which the money committed rises only as the evidence does, and in which the transition that matters is paper to small live, because that is where slippage, fills, and operational reality first get a vote. A pipeline that jumps from backtest to full allocation has no stage at which reality can object cheaply.

The staffing table is the scaling answer stated as a budget. **A single sleeve needs 56 tested ideas a year, roughly one researcher, $204,000, and $6.4 million of AUM to fund the research alone** — before the $450,000 wrapper from [the first lesson](01-capital-fund-structures-fees.md). **Six sleeves need 333 ideas a year, 5.6 researchers, $1.22 million, and $38 million of AUM.** Set against that lesson's $14.1 million break-even for a trend book, the shape of the business becomes clear: a small fund can afford operations or research, not both, and the founder is the research department until roughly $20 million.

The sensitivity worth checking is sleeve life. It divides directly into the intake requirement, so a book whose strategies decay in two years rather than four needs twice the research organization to stand still. That is the quantitative form of a familiar observation — that crowded, fast-decaying edges are a worse business than slow ones at the same Sharpe — and it belongs in the decision about which strategy types to pursue, not in the post-mortem.

## Hiring for work you cannot supervise

Quantitative research is unusually hard to supervise, because a bad result and a bad process look identical from outside and both take months to distinguish. Hiring is therefore mostly about work samples, and the sample should look like the job.

| Role | What they own | The work sample that predicts it |
|---|---|---|
| Researcher | Hypotheses, tests, kill decisions | Hand them a notebook with a deliberate lookahead bug and a flattering result; ask what they conclude |
| Engineer | The platform, data, execution path | A failing test in an unfamiliar codebase and a fixed hour |
| Ops / trading | The daily calendar, breaks, NAV, brokers | The break list from lesson three: classify, prioritize, and say what blocks trading |

The researcher sample is the one worth building carefully, and the [Part IX](../part-09-software-engineering/01-git-and-code-review.md) review material supplies it directly: a notebook whose backtest shows a Sharpe of 2.4 because a feature is computed with `shift(-1)` somewhere unobvious. What you are testing is not whether they find the bug within the hour — plenty of good people do not. It is what they do with a Sharpe of 2.4 *before* they find it. A candidate whose first move is suspicion, who asks what would have to be true for the number to be real, is demonstrating the only habit that matters. A candidate who starts optimizing the parameters has told you as much.

Avoid the interview that is a mathematics examination. It selects for a skill that correlates weakly with research judgement and screens out experienced practitioners who have not derived an Itô expansion since graduate school. Ask instead for a strategy the candidate abandoned, and why — a question that is almost impossible to answer well without having actually done the work.

Onboarding is where the registry pays a second time. A new researcher who reads two years of `kill_reason` entries in their first week arrives at the frontier instead of rediscovering it, and the exercise doubles as the most honest possible description of what the firm has learned. The documentation that survives departures is not architecture diagrams; it is the record of what was tried and what happened.

## One book to several

Running several sleeves is not running one sleeve more times. Three things change, and each has a structural answer.

**Allocation becomes a decision that must be made by someone.** With one book, the sizing question is [Part VIII's](../part-08-portfolio-management/02-kelly-vol-targeting-leverage.md) and it is answered by volatility targeting. With six, the risk budget is a scarce resource competing between sleeve owners who each believe theirs is underweighted. The answer that works is a stated, written allocation rule — marginal contribution to portfolio risk, with a floor and a cap per sleeve — applied mechanically, and revisited quarterly rather than continuously. An allocation that can be argued for in the moment will be.

**Shared infrastructure becomes the constraint and the advantage.** One data pipeline, one execution path, one reconciliation, one risk system, one deployment process — this is the entire economic argument for a multi-strategy book, because the marginal operational cost of sleeve seven is close to zero if and only if the first six share a platform. It is also the argument for everything in [Part IX](../part-09-software-engineering/03-package-structure-config-di.md): a platform with real seams supports six strategies, and a pile of per-strategy scripts supports one.

**Oversight has to be built rather than felt.** With one book the founder knows when something is wrong. With six, they do not, and the substitute is a small number of aggregate controls: a firm-wide risk limit that no sleeve can breach individually, a daily attribution that reconciles each sleeve's P&L to its model's expectation, and an automatic reduction rule for a sleeve at a stated drawdown. The last one is worth writing down before it is needed, because the conversation about cutting a colleague's allocation in the middle of their drawdown is exactly the conversation that does not go well improvised.

The order in which capacity is consumed is worth a final thought. Adding sleeves buys diversification, and diversification is what lets the book carry more capital at the same risk — which is the only lever that grows the business without needing a better strategy. That is the actual reason to run a research pipeline: not because the current sleeve is bad, but because six mediocre uncorrelated sleeves carry more money at a higher Sharpe than one good one, and money is what pays for the next six.

!!! abstract "Key takeaways"
    - A gate is only real if it has a written evidence standard and a record of rejections. The most under-used gate is intake — one paragraph naming who is on the other side of the trade — and the most often missing is correlation with the existing book.
    - The standard screen fails badly. Of 1000 ideas in a generous world, a Sharpe-1.0 screen on three years of data kept **53, only 32% real**, promising **1.25** and delivering **0.19** — an **85% haircut**.
    - Tightening the screen to 1.50 still left a **67% haircut**, and the statistically correct threshold for 1000 trials — Sharpe **2.15** — **rejected everything**. Three years of data cannot support a thousand looks.
    - The fix is data, not strictness: the **same lenient screen on ten years kept 9 ideas, 100% of them real**, delivering the same 0.55 out of sample as the strict three-year gate but without the noise.
    - Every correction depends on N, the number of ideas actually tested including abandoned variants. A firm without a registry cannot compute its own N and therefore defaults to the failing screen. That is the strongest argument for the registry, ahead of institutional memory.
    - Composed pass rates give **222 ideas per allocation**. A researcher who has tried thirty things and found nothing is one seventh of the way to a result.
    - Scaling is a budget: **one sleeve needs 56 ideas a year and $6.4M of AUM to fund the research; six sleeves need 333 ideas, 5.6 researchers, and $38M** — against the $14.1M break-even from lesson one. Below roughly $20 million the founder is the research department.
    - Sleeve life divides straight into the intake requirement, so a book of fast-decaying edges needs twice the research organization to stand still.
    - Hire on work samples that look like the job. For researchers, the signal is what a candidate does with a Sharpe of 2.4 *before* finding the lookahead bug.
    - Several sleeves need a written allocation rule applied mechanically, one shared platform, and aggregate controls — including a drawdown reduction rule agreed before anyone is in a drawdown.

## Where this goes next

This is the end of the course core. Part I established what the instruments and participants are, Parts II and III built the tools, Part IV built strategies and then broke most of them honestly, Part V built an engine that could be trusted, Part VI put it into production, Part VII established where machine learning helps and mostly where it does not, Part VIII turned a set of strategies into a portfolio, Part IX made the codebase something a professional would sign their name to, and Part X put a business around it.

The through-line is worth stating once more, because it is the only thing here that generalizes: nearly every result in this course that mattered was a number that contradicted the argument for producing it. The engine that disagreed with the vectorized backtest, the machine-learning models that lost to a linear baseline, the optimizer beaten by equal weights, the conflation policy that looked best on every operational metric and changed the strategy, the fee path dependence that turned out to live in the management fee, and now a research screen that promises 1.25 and delivers 0.19. The discipline the course is actually teaching is the one that produces those numbers and then reads them.

Where to go from here depends on what you are building. The [optional advanced modules](../advanced/index.md) go deeper into thirteen specific areas — execution and market impact, filtering, options, market making, alternative data, distributed and GPU computing — and each is self-contained, with its own prerequisites; nobody needs all of them. The [appendix](../appendix/index.md) is the mathematics reference the core lessons draw on, organized to be consulted rather than read through. Both are there for when a specific piece of work demands them, which is the only good reason to read either.

The more useful next step is not reading. It is running the pipeline in this lesson on one real idea, keeping the registry entry honest, and letting the gate say no.
