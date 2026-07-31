# Investor Reporting and Due Diligence

[Capital, Fund Structures, and Fees](01-capital-fund-structures-fees.md) ended on a number you have to raise — eleven to twenty-three million dollars of other people's money before the founder clears $300,000. Nobody wires that against a fee schedule. They wire it against a track record they believe and an operation they have inspected, and the inspection is conducted by someone whose job is to find the reason to decline.

This lesson builds the monthly report, then attacks it. The report itself contains one number most managers omit and every allocator computes: the book below spent **55 of its 96 months under water**, against a maximum drawdown of 21% that sounds far milder. The attack is the more useful half. Four presentation choices — each individually defensible, none of them a false statement — take the same underlying track from a **net Sharpe of 0.70 to 1.39** and its CAGR from **6.92% to 14.65%**. The largest single step is not the one usually warned about, and the one usually warned about does nothing at all here.

!!! note "Scope"
    Both blocks are self-contained and seeded, and they share the same underlying track: the "as traded" line of the second block is the "net of fees" line of the first. GIPS is referred to as a standard to be aware of rather than summarized; verifying compliance is an engagement for a verifier, not a paragraph in a lesson.

## An allocator reads the report for what is missing

Reporting is a cadence, not a document. Several run in parallel, and they fail differently: a late flash estimate reads as an operational problem, a late annual audit reads as an existential one.

| Cadence | Sent | Contents | What lateness signals |
|---|---|---|---|
| Flash estimate | 1–3 business days after month end | One number, clearly marked estimate | Your NAV process is manual |
| Monthly letter | 10–15 business days | Return, exposures, risk, attribution, commentary | Nothing yet; twice in a row, something |
| Quarterly call | Within a month of quarter end | Positioning, capacity, personnel, terms | Avoidance |
| Audited annual | 90–120 days after year end | Audited financials, GIPS-aware presentation | An unresolved disagreement with the auditor |

The monthly letter carries the weight. What belongs in it is not a matter of taste — an allocator's model wants specific fields, and anything absent gets requested by email, which costs you the credibility of having volunteered it.

| Section | Must contain | Commonly and wrongly omitted |
|---|---|---|
| Performance | Monthly and since-inception, **net**, against a stated benchmark | Gross alongside net, with the fee bridge shown |
| Exposure | Gross, net, by sleeve and asset class, month-end and average | Average — month-end alone hides intra-month leverage |
| Risk | Volatility, max drawdown, current drawdown, VaR or stress numbers | Time under water, and the largest single position |
| Attribution | Contribution by sleeve, reconciling to the total | Costs as their own line rather than netted into the sleeves |
| Capacity | AUM, subscriptions, redemptions, remaining capacity | Redemptions — reporting only subscriptions is noticed |
| Commentary | What happened and whether it was expected | What was expected and did not happen |

## The report, generated

The book here is the two-sleeve structure the course has used throughout: a carry sleeve funded from the start and a dispersion sleeve that ran for five years, gave most of it back, and was closed at the end of 2022.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(10)


def draws(n, mean, vol):
    """De-meaned, so the block reflects its stated assumptions and not its luck."""
    x = rng.normal(0.0, vol, n)
    return x + mean - x.mean()


# Eight years of a two-sleeve book. Carry was funded from the start; dispersion
# ran for five years, gave back most of it, and was closed at the end of 2022.
idx = pd.period_range("2018-01", periods=96, freq="M")
carry, disp = draws(96, 0.0100, 0.038), draws(96, 0.0035, 0.052)
disp[36:60] -= 0.009
gross = pd.Series(np.where(np.arange(96) < 60, 0.5 * carry + 0.5 * disp, carry), index=idx)


def net_of_fees(g, M=0.02, P=0.20):
    """Monthly management accrual, annual performance crystallization over a HWM."""
    nav, hwm, out = 1.0, 1.0, []
    for i, r in enumerate(g):
        start = nav
        nav = nav * (1 + r) - M / 12 * nav
        if (i + 1) % 12 == 0:
            nav -= P * max(nav - hwm, 0.0)
            hwm = max(hwm, nav)
        out.append(nav / start - 1)
    return pd.Series(out, index=g.index)


def stats(r):
    cagr = float((1 + r).prod()) ** (12 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)      # excess over a zero cash rate
    curve = (1 + r).cumprod()
    return cagr, sharpe, float((curve / curve.cummax() - 1).min())


net = net_of_fees(gross)
wide = (net.to_frame("r").assign(y=net.index.year, m=net.index.month)
        .pivot(index="y", columns="m", values="r").mul(100))
wide["YEAR"] = ((1 + net).groupby(net.index.year).prod() - 1).mul(100)
wide.index.name = wide.columns.name = None
print("  monthly returns net of all fees, per cent")
print(wide.to_string(float_format=lambda v: f"{v:6.2f}"))
print(f"\n  {'':18s} {'CAGR':>7s} {'Sharpe':>7s} {'max DD':>8s}")
for label, r in (("gross of fees", gross), ("net of fees", net)):
    c, s, d = stats(r)
    print(f"  {label:18s} {c:7.2%} {s:7.2f} {d:8.2%}")
curve = (1 + net).cumprod()
under = (curve < curve.cummax()).astype(int)
longest = max((g.sum() for _, g in under.groupby((under == 0).cumsum())), default=0)
print(f"  positive months {int((net > 0).sum())}/{len(net)}, "
      f"best {net.max():+.2%}, worst {net.min():+.2%}, "
      f"longest time under water {int(longest)} months")
print("  sleeves: carry (live since 2018-01), dispersion (2018-01 to 2022-12, closed)")
# =>   monthly returns net of all fees, per cent
#              1      2      3      4      5      6      7      8      9     10     11     12   YEAR
#    2018  -0.59   1.36   4.76   5.01   0.74   4.35   3.64  -0.49   2.36   3.46  -1.25  -1.99  23.15
#    2019   0.28   1.70  -3.21   4.96   3.49   0.83   0.22  -1.27   3.01  -2.23   4.58  -0.24  12.40
#    2020   5.71  -3.72  -3.66   5.47   3.42   2.92   4.00  -4.07  -1.31  -1.64  -2.50  -0.79   3.13
#    2021  -1.71  -0.73   1.24  -2.36  -0.37   1.78   2.08   3.23  -2.14   4.09   2.68  -3.95   3.53
#    2022  -1.77   0.28  -1.55   5.65  -0.79   4.40  -4.20  -4.10  -2.85  -0.97  -0.08  -0.59  -6.84
#    2023  -2.06  -2.27   1.28   4.29  -3.78  -6.18   0.60   0.27  -1.73   4.05   4.78  -0.42  -1.79
#    2024   0.42   3.89   0.09  -3.61   1.47  -1.71   2.96  -0.19   1.01   3.70   1.25  -0.68   8.65
#    2025  -2.10   8.27   5.93  -3.99  -2.27   3.94  -0.51   1.78   4.82   2.21   1.23  -3.33  16.26
#
#                            CAGR  Sharpe   max DD
#      gross of fees       10.75%    1.05  -18.54%
#      net of fees          6.92%    0.70  -21.41%
#      positive months 51/96, best +8.27%, worst -6.18%, longest time under water 55 months
#      sleeves: carry (live since 2018-01), dispersion (2018-01 to 2022-12, closed)
```

This is a fundable record and not an exciting one. **6.92% net at a Sharpe of 0.70**, which is roughly what an honest systematic book looks like once fees are taken out — and the fees are not a rounding error: gross 10.75% becomes net 6.92%, so **the fee structure consumed 3.83 percentage points, or 36% of the gross return**. That bridge belongs in the letter. An allocator who has to derive it will assume you hoped they would not.

The line that decides the meeting is the last risk number. Maximum drawdown of **21.41%** is a single bad moment and reads as survivable. **55 of 96 months under water** — 57% of the record — is the experience of holding it, and it is what the allocator's investment committee actually has to sit through. The two numbers describe the same four years, 2020 through 2023, where the yearly column reads +3.13%, +3.53%, −6.84%, −1.79%: no catastrophe, just four years of going nowhere while fees accrued. Most managers report the drawdown and not the time. Reporting the time first is cheap credibility, because the allocator computes it either way.

## Track records are presentations, not facts

Now take the identical underlying returns and present them the way a manager under fundraising pressure presents them. Each step below is a decision someone can defend in a sentence, and every resulting number is arithmetically correct.

```python
import numpy as np

rng = np.random.default_rng(10)


def draws(n, mean, vol):
    x = rng.normal(0.0, vol, n)
    return x + mean - x.mean()


carry, disp = draws(96, 0.0100, 0.038), draws(96, 0.0035, 0.052)
disp[36:60] -= 0.009                                # the run that got it closed
sim = draws(24, 0.0180, 0.030)                      # backtested, in sample, never traded
composite = np.where(np.arange(96) < 60, 0.5 * carry + 0.5 * disp, carry)


def net(g, M=0.02, P=0.20):
    nav, hwm, out = 1.0, 1.0, []
    for i, r in enumerate(g):
        start = nav
        nav = nav * (1 + r) - M / 12 * nav
        if (i + 1) % 12 == 0:
            nav -= P * max(nav - hwm, 0.0)
            hwm = max(hwm, nav)
        out.append(nav / start - 1)
    return np.array(out)


def stats(r):
    cagr = float(np.prod(1 + r)) ** (12 / len(r)) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    curve = np.cumprod(1 + r)
    return len(r), cagr, sharpe, float((curve / np.maximum.accumulate(curve) - 1).min())


traded = net(composite)
# "The record begins when the strategy reached its current form" is, in practice,
# a search over start dates. Do the search explicitly, keeping five years of record.
best = max(range(37), key=lambda k: stats(traded[k:])[2])

presentations = [
    ("as traded, net, both sleeves", traded),
    (f"+ record starts at month {best + 1}", traded[best:]),
    ("+ closed sleeve removed", net(carry)[best:]),
    ("+ shown gross of fees", carry[best:]),
    ("+ backtest period backfilled", np.concatenate([sim, carry[best:]])),
]

print(f"  {'presented as':32s} {'months':>7s} {'CAGR':>8s} {'Sharpe':>7s} {'max DD':>8s}")
for label, r in presentations:
    n, c, s, d = stats(r)
    print(f"  {label:32s} {n:7d} {c:8.2%} {s:7.2f} {d:8.2%}")
print("  every line is arithmetically correct and none contains a false statement")
# =>   presented as                      months     CAGR  Sharpe   max DD
#      as traded, net, both sleeves          96    6.92%    0.70  -21.41%
#      + record starts at month 2            95    7.08%    0.71  -21.41%
#      + closed sleeve removed               95    8.36%    0.78  -16.86%
#      + shown gross of fees                 95   12.51%    1.15  -14.56%
#      + backtest period backfilled         119   14.65%    1.39  -14.56%
#      every line is arithmetically correct and none contains a false statement
```

End to end the record goes from **6.92% at Sharpe 0.70 to 14.65% at Sharpe 1.39**, with maximum drawdown improving from −21.41% to −14.56%. Nothing was fabricated. Each line is a different answer to the question "what is your track record", and a manager could give any of them under oath.

The step sizes are not where the folklore puts them. **The single largest move is gross-for-net, worth 0.37 of Sharpe and 4.15 percentage points of CAGR** — and it is the easiest one to commit by accident, because the research pipeline naturally produces gross numbers and somebody has to remember to run them through the fee model before they reach a page. It is also the easiest to catch: ask which fee schedule the numbers are net of, and a manager who has to go and check has answered.

**The cherry-picked start date did essentially nothing: 0.70 to 0.71, one basis point of Sharpe, and it was the best of 37 candidate start dates.** That is a real result and worth understanding rather than dismissing. This book's bad stretch is in the middle of its life, 2021 through 2023, so no start date excludes it while leaving five years of record. Start-date selection is devastating on a track whose worst period is at the beginning and useless on one whose worst period is in the middle — which is precisely why you check it rather than assume it.

Removing the closed sleeve moved Sharpe modestly, 0.71 to 0.78, but moved **maximum drawdown from −21.41% to −16.86%** — nearly five points, on the number most allocators size against. Survivorship in a composite is usually discussed as a return effect; measured here, it is mostly a risk effect. And the backfill added **0.24 of Sharpe from 24 months that were never traded** — the period the model was fitted on, which is to say the period it was guaranteed to do well in.

!!! warning "The disclosure is not the defence"
    Every step above can be disclosed in a footnote, and disclosing it is necessary. It is not sufficient. A tearsheet whose headline Sharpe is 1.39 and whose footnote explains that 24 of the 119 months are simulated has still put 1.39 in the reader's head. If simulated results appear at all, the live-only figures belong in the same table at the same size — which is the substance of what GIPS is trying to enforce, and the reason a manager who intends to raise institutional money engages a verifier early rather than reconstructing composites under diligence.

## The ODD is a test of your operation, not your returns

Operational due diligence is a separate process from investment due diligence, often run by a separate team with a veto and no interest in your Sharpe. Its subject is whether the money can go missing.

| Area | What is asked for | What is actually being tested |
|---|---|---|
| Service providers | Administrator, auditor, prime broker, counsel — names and contacts | That they exist, are independent, and confirm you are a client |
| Valuation | Pricing policy, who signs the NAV, exception handling | That the manager cannot mark their own book |
| Cash controls | Who can move money, dual authorization, wire callbacks | Segregation of duties in a firm too small to have any |
| Technology | Source control, backups, disaster recovery, access | Whether a laptop failure ends the business |
| Compliance | Registration status, personal trading policy, records | That obligations were identified, not that they are onerous |
| Key person | Who trades if you are unavailable for three months | Whether the business is a person |
| Terms | Gates, side letters, lock-ups, expense allocation | Whether other investors got better terms |

The red flags that end processes are mundane and mostly structural: the manager's brother-in-law is the administrator; the auditor is a firm nobody has heard of; performance is calculated by the manager from the manager's own records; one person can both instruct and approve a wire; there is no written pricing policy for the one illiquid position; a side letter grants another investor a better redemption right; the answer to the key-person question is silence.

Small managers cannot solve all of these with headcount, and allocators know it. What they check for is that you identified the gap and compensated for it deliberately — a third-party administrator instead of internal NAV, a dual-authorization wire policy with a named second signatory, a documented three-month contingency naming an actual person. A compensating control that is written down and tested beats an unmitigated gap by more than it costs.

## Bad months are a scheduled event

You will have a bad month, and the report for it is the most consequential one you write. The rule is that the allocator hears it from you first, in a channel you chose, before the monthly letter — and that the note answers three questions in order: what the number is, whether the loss came from the strategy behaving as designed or from something breaking, and what if anything changes.

The second question is the whole letter. A drawdown from a trend book during a sharp reversal is the strategy working; saying so plainly, and pointing at the same explanation you gave in an earlier letter for why that risk exists, builds the position that carries you through the next one. A loss from a bug, a bad fill, or a limit that did not fire is a different letter entirely, and it must say what the control failure was and what changed. Managers who blur these two are found out, because allocators keep the letters and compare them.

The one thing that reliably ends a relationship is a bad month that arrives with an explanation the investor has never heard before — a risk that was never in a letter, an exposure never in a report, a sleeve nobody knew was live. That is not a performance problem. It is a disclosure problem, and it is why the exposure table in the first section is worth writing carefully every month when nothing is happening.

!!! abstract "Key takeaways"
    - The book reported **6.92% net at Sharpe 0.70** from **10.75% gross** — the fee structure took **3.83 percentage points, 36% of the gross return**. Publish that bridge; the allocator computes it regardless.
    - Maximum drawdown of **21.41%** understates the experience of holding it. The same record spent **55 of 96 months under water**, and time under water is what an investment committee actually sits through.
    - Four defensible presentation choices took the same returns from **Sharpe 0.70 to 1.39 and CAGR 6.92% to 14.65%**, with no false statement anywhere.
    - The biggest single step was **gross-for-net, worth 0.37 of Sharpe**, and it is the one most easily committed by accident because research code produces gross numbers by default.
    - **Cherry-picking the start date bought 0.01 of Sharpe from the best of 37 candidates** — useless here, because this book's bad years are in the middle. Whether a given distortion works depends on the shape of the record, which is why each is checked rather than assumed.
    - Removing the closed sleeve was mostly a *risk* effect: Sharpe rose 0.07, but maximum drawdown improved **from −21.41% to −16.86%**, on the number allocators size against.
    - Operational due diligence asks whether the money can go missing, not whether the strategy is good. Small managers pass it with documented compensating controls, not with headcount.

## Where this goes next

The diligence questionnaire in this lesson asks who signs the NAV, whether anyone can move cash alone, and how a pricing exception is handled. Those are not questions about policy documents. They are questions about a daily process that either runs or does not, and the answers an allocator finds convincing are the ones backed by evidence that it ran yesterday.

[Operations, Compliance, and Tax](03-operations-compliance-tax.md) builds that process: reconciling the book against the broker and classifying the breaks, calculating a NAV that survives a stale price, and mapping the compliance and tax obligations that arrive with each increment of other people's money.
