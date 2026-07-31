# Operations, Compliance, and Tax

[Investor Reporting and Due Diligence](02-investor-reporting-due-diligence.md) ended with a questionnaire asking who signs the NAV, whether one person can move cash alone, and how a pricing exception is handled. Those are questions about a daily process, and the only convincing answer is evidence that it ran yesterday. This lesson builds the process.

Two numbers set the stakes. A reconciliation of forty fills against the broker's file finds **seven breaks in a file with exactly the right number of rows** — a row-count check passes cleanly while 550 shares of position sit unaccounted for. And a single stale price on 7% of the book overstates NAV by **$48,918**, which transfers **$5,048.55** away from an incoming subscriber and crystallizes **$12,229.58 of performance fee that was never earned** — the larger of the two errors landing with the party least motivated to catch it.

!!! warning "Orientation, not advice"
    The compliance and tax sections describe **categories of obligation and the questions to ask**, not the rules of any jurisdiction. Thresholds, exemptions, entity treatments, and filing deadlines differ by country and change; nothing here is legal, tax, or accounting advice, and the correct use of these sections is to arrive at a professional's office already knowing which questions apply to you. Both code blocks are self-contained and seeded.

## Reconciliation is the only control that sees everything

Every other control checks one thing: a risk limit checks exposure, a kill switch checks a threshold, a log checks what you thought to log. Reconciliation checks the whole book against an independent record, and it is the only place where a discrepancy you did not anticipate can surface. It runs daily, before anything else, and its output is a list of breaks that are individually explained or escalated — never a green tick.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)

# One trading day of fills as our own system recorded them.
n = 40
book = pd.DataFrame({
    "oid": [f"o{i:03d}" for i in range(n)],
    "sym": rng.choice(["SPY", "IEF", "GLD", "EFA"], n),
    "qty": rng.choice([-400, -200, -100, 100, 200, 400], n),
    "px": np.round(rng.uniform(80, 420, n), 2),
    "fee": 0.65,
    "day": 0,
})

# The broker's file for the same day, carrying the six breaks that actually happen.
bro = book[book.oid != "o007"].copy()                         # never reached the broker
bro.loc[bro.oid == "o013", "qty"] //= 2                       # partial fill we booked in full
bro.loc[bro.oid == "o021", "px"] += 0.03                      # filled off our recorded price
bro.loc[bro.oid == "o028", "fee"] = 2.15                      # exchange fee we do not model
bro.loc[bro.oid.isin(["o033", "o034"]), "day"] = 1            # booked after our cut-off
ghost = book[book.oid == "o005"].assign(oid="o099", qty=300)  # a fill we have no record of
bro = pd.concat([bro, ghost], ignore_index=True)


def classify(r):
    if pd.isna(r.qty_b):
        return "missing at broker"
    if pd.isna(r.qty_a):
        return "unknown to us"
    if r.qty_a != r.qty_b:
        return "quantity mismatch"
    if r.px_a != r.px_b:
        return "price mismatch"
    if r.fee_a != r.fee_b:
        return "fee mismatch"
    if r.day_a != r.day_b:
        return "timing"
    return "matched"


def cash(qty, px, fee):
    """Cash leaving the account: negative to buy, positive to sell, fees always out."""
    return -(qty * px) - fee


m = book.merge(bro, on="oid", how="outer", suffixes=("_a", "_b"))
m["break"] = m.apply(classify, axis=1)
m["shares"] = m.qty_a.fillna(0) - m.qty_b.fillna(0)
m["cash"] = (cash(m.qty_a.fillna(0), m.px_a.fillna(0), m.fee_a.fillna(0))
             - cash(m.qty_b.fillna(0), m.px_b.fillna(0), m.fee_b.fillna(0)))

breaks = m[m["break"] != "matched"]
out = (breaks.groupby("break")
       .agg(fills=("oid", "size"), shares=("shares", "sum"), cash=("cash", "sum"))
       .sort_values("cash", key=abs, ascending=False))
out["clears itself"] = ["yes, at T+1" if b == "timing" else "no" for b in out.index]
out.index.name = None

print(f"  {len(book)} fills booked, {len(bro)} on the broker file, "
      f"{int((m['break'] == 'matched').sum())} matched, {len(breaks)} broken")
print(out.to_string(float_format=lambda v: f"{v:11.2f}"))
print(f"  our records minus the broker's: {out.shares.sum():+.0f} shares, "
      f"{out.cash.sum():+,.2f} cash")
fee_break = float(out.loc["fee mismatch", "cash"])
print(f"  the fee break is {fee_break:+.2f} today and {abs(fee_break) * 252:,.0f} a year "
      f"if it recurs daily, and it means the cost model is wrong")
# =>   40 fills booked, 40 on the broker file, 34 matched, 7 broken
#                       fills      shares        cash clears itself
#    unknown to us          1     -300.00   101388.65            no
#    missing at broker      1     -200.00    22957.35            no
#    quantity mismatch      1      -50.00     8963.50            no
#    price mismatch         1        0.00       -3.00            no
#    fee mismatch           1        0.00        1.50            no
#    timing                 2        0.00        0.00   yes, at T+1
#      our records minus the broker's: -550 shares, +133,308.00 cash
#      the fee break is +1.50 today and 378 a year if it recurs daily, and it means the cost model is wrong
```

Start with what did not catch it. **Forty fills booked, forty on the broker file** — the row counts agree exactly, because one fill vanished and one appeared. Any check that compares file sizes, or counts trades, or asserts that the day's activity "looks right", passes. The breaks are only visible to a join on the order identifier, which is the argument for having one: an identifier the broker echoes back is what makes reconciliation possible at all, and it is why [Part VI's](../part-06-live-infrastructure/05-resilience-and-risk-controls.md) client order IDs were worth the trouble.

The six classes divide into three very different problems. **The two position breaks — one fill missing at the broker, one fill unknown to us, together 500 shares and $124,346** — are the emergency. Your position is wrong, which means your risk numbers are wrong, your sizing is wrong, and tomorrow's orders are computed from a book that does not exist. The "unknown to us" break is the worse of the two: a fill nobody in your system asked for is either a fat-fingered manual trade, a duplicate submission, or someone else's trade booked to your account, and all three need an answer before the next open.

The **quantity mismatch — a partial fill you booked in full, 50 shares and $8,963** — is the most common break in practice and the most dangerous to automate away, because the natural fix ("trust the broker") is correct for the position and wrong for the diagnosis: something in your fill handling ignored a partial, and it will do it again.

The **price and fee breaks are trivial money and important information**. Three dollars and one dollar fifty. The fee break in particular is an exchange fee the cost model does not know about; **at one fill in forty every day it is $378 a year**, which nobody would chase, but it means the transaction-cost assumptions in every backtest are slightly optimistic in a way that compounds with turnover. Chase the discrepancy, not the dollars.

**Timing breaks are the ones that ruin the control.** Two fills booked after the cut-off, zero shares, zero cash, and they clear themselves at T+1 without anyone doing anything. They are also the most numerous class in a real operation, and a break list dominated by them trains whoever reads it to scroll past. The fix is structural rather than diligent: classify timing breaks automatically and age them, so a "timing" break that is still open at T+2 escalates as a real break — because at that point it is one.

!!! note "Aged, not just counted"
    A break list is a queue with an age column, not a daily count. The useful policy is a maximum age per class: timing breaks die at T+1, price and fee breaks are explained within the week, and any position break blocks trading until it is resolved. An allocator's operational reviewer will ask to see the oldest open break — which is a question about the queue, not about yesterday.

## NAV is a number somebody has to defend

Reconciliation establishes what you own. NAV puts a price on it, and the difference between those two sentences is where most valuation disputes live. A pricing policy answers three questions in advance: which source is primary for each instrument, what happens when the source does not publish, and who is allowed to override. The third one is the reason a manager should not strike their own NAV — and the block below is why.

```python
import pandas as pd

UNITS, CASH, HWM_PER_UNIT = 100_000.0, 1_250_000.0, 180.00
MGMT_MO, ADMIN_MO, AUDIT_MO, PERF = 0.02 / 12, 4_500.0, 3_000.0, 0.20

# Month-end holdings. The vendor published four of the five prices; the fifth
# is an off-the-run credit ETF that did not print, so the pricing policy's
# fallback carried yesterday's close forward.
pos = pd.DataFrame({
    "sym": ["SPY", "IEF", "GLD", "EFA", "XCB"],
    "qty": [12_000, 40_000, 15_000, 30_000, 25_000],
    "used": [585.20, 94.15, 248.60, 82.40, 61.00],      # what the NAV was struck on
    "true": [585.20, 94.15, 248.60, 82.40, 58.55],      # what it was actually worth
})


def nav(prices):
    gav = float((pos.qty * prices).sum()) + CASH
    expenses = MGMT_MO * gav + ADMIN_MO + AUDIT_MO
    n = gav - expenses
    perf = PERF * max(n - HWM_PER_UNIT * UNITS, 0.0)
    return gav, expenses, perf, n - perf


struck = nav(pos.used)
correct = nav(pos["true"])
labels = ["gross asset value", "accrued expenses", "performance fee", "net asset value"]
print(f"  {'':20s} {'as struck':>14s} {'corrected':>14s} {'difference':>12s}")
for label, a, b in zip(labels, struck, correct):
    print(f"  {label:20s} {a:14,.2f} {b:14,.2f} {a - b:12,.2f}")

pps, ppc = struck[3] / UNITS, correct[3] / UNITS
print(f"  {'NAV per unit':20s} {pps:14,.4f} {ppc:14,.4f} {pps - ppc:12,.4f}")

# A subscription settles at the published month-end price per unit.
SUB = 2_000_000.0
issued = SUB / pps
print(f"\n  a {SUB:,.0f} subscription buys {issued:,.2f} units at the published "
      f"{pps:,.4f}")
print(f"  those units are worth {issued * ppc:,.2f} on corrected prices: "
      f"the subscriber is out {SUB - issued * ppc:,.2f}")
print(f"  the existing holders are up by the same amount, and nobody sends it back")
print(f"  the fee crystallized on the wrong number is over by "
      f"{struck[2] - correct[2]:,.2f}")
# =>                             as struck      corrected   difference
#      gross asset value     19,764,400.00  19,703,150.00    61,250.00
#      accrued expenses          40,440.67      40,338.58       102.08
#      performance fee          344,791.87     332,562.28    12,229.58
#      net asset value       19,379,167.47  19,330,249.13    48,918.33
#      NAV per unit               193.7917       193.3025       0.4892
#
#      a 2,000,000 subscription buys 10,320.36 units at the published 193.7917
#      those units are worth 1,994,951.45 on corrected prices: the subscriber is out 5,048.55
#      the existing holders are up by the same amount, and nobody sends it back
#      the fee crystallized on the wrong number is over by 12,229.58
```

The error is small and entirely ordinary: one position out of five, 7% of the book, priced 4% high because the vendor did not publish and the policy said carry yesterday forward. It overstates gross assets by **$61,250 — 0.31% of the fund** — which would round to nothing in a monthly letter.

It does not round to nothing anywhere else. The subscriber who wired $2,000,000 that day received **10,320.36 units worth $1,994,951.45**, so **$5,048.55 moved from them to the existing holders** at the moment of subscription. This is dilution, it is the thing equalization and series accounting exist to prevent, and the important property is that it is *permanent*: when the price corrects tomorrow the units are simply worth what they are worth, and no mechanism sends the money back.

The larger number is the last one. **$12,229.58 of performance fee crystallized on a NAV that was never real** — two and a half times the subscriber's loss, and it is paid to the manager. That is the whole argument for independent valuation stated arithmetically: the person best placed to notice a stale price is the person being paid on it, and no amount of integrity makes that a good control design. An administrator who strikes the NAV, a written policy that names the fallback before the day it is needed, and a documented exception log are what convert this from a judgement call into a process.

Note also the accruals, which are the boring half of NAV and the half auditors query. Expenses of **$40,440.67** for the month are a management-fee accrual plus a fixed administration cost plus one twelfth of an annual audit fee that has not yet been invoiced. Accruing the audit monthly rather than expensing it when the bill arrives is what stops December's NAV from taking a visible step down for reasons unrelated to trading.

## The broker is a single point of failure you selected

Broker selection is usually argued on commission rates, which are the smallest of the differences between one and another.

| Dimension | What to ask | Why it decides |
|---|---|---|
| Custody | Are assets segregated, and under whose name? | This is the question that matters when the broker fails |
| Financing | Rate schedule, and how it moves with your balance | Often larger than commission for a levered book |
| Borrow | Availability, recall history, rate stability | A recall can close a position you were relying on |
| Reporting | Machine-readable fills, positions, and fees, same day | This is what makes the reconciliation above possible |
| API | Rate limits, order types, sandbox that behaves like production | The subject of most of Part VI |
| Redundancy | Can positions be moved, and how quickly? | A second relationship is cheap; discovering you need one is not |

A second broker relationship, even a dormant one with a small balance, is the cheapest insurance in the business. It gives you an independent price source when the primary vendor fails, somewhere to move if a credit event or a compliance dispute freezes the primary, and negotiating leverage that a single-broker manager does not have. Managers who skip it usually cite the operational overhead of a second reconciliation — which is the same overhead, run twice, and the reconciliation is already built.

## Obligations arrive with other people's money

Every regime differs, but the *shape* is consistent: obligations step up at thresholds, and the thresholds are about whose money you manage and how you got it, not how much you make.

| Trigger | Obligations that typically appear | The question to ask |
|---|---|---|
| Trading only your own capital | Tax filing; exchange and venue rules | Am I inadvertently advising anyone? |
| First outside investor | Registration or an exemption; advisory duties; recordkeeping | Which exemption, and what conditions keep it? |
| Pooling investors into a vehicle | Fund registration or private-placement exemption; offering documents; investor eligibility tests | Who may I accept, and what may I say to them? |
| Crossing an AUM or investor-count threshold | Full registration, compliance officer, formal policies, examinations | What is the threshold and how long is the runway? |
| Marketing across borders | Local registration or reverse-solicitation limits | Is a website "marketing" in that jurisdiction? |
| Trading certain instruments | Position-limit reporting, large-trader IDs, short-sale disclosure | Which reports are triggered by size rather than by entity? |

Two areas catch systematic managers specifically. The first is **recordkeeping**: retention periods commonly run to five or seven years and usually cover business communications as well as trade records — which for a quantitative firm means research notebooks, model changes, and the chat where somebody said "just turn the limit off for today". This is where [Part IX's](../part-09-software-engineering/01-git-and-code-review.md) reproducible-from-a-commit-hash discipline stops being an engineering preference and starts being a retention policy that happens to already work.

The second is **marketing rules**. Performance advertising is regulated almost everywhere, and the presentation choices measured in the previous lesson are exactly what those rules address: whether net figures must accompany gross, how simulated results must be labelled and positioned, whether a testimonial is permitted, and what a "track record" may include when the manager changes firms. A tearsheet is a marketing document. Have someone qualified read the first one.

## Tax is a set of questions, not a set of answers

Tax treatment is the most jurisdiction-specific material in this course and the most consequential to get wrong, because it is assessed years later with interest. The useful preparation is knowing which questions apply.

- **Entity treatment.** Is the vehicle taxed itself, or does it pass through to holders? Pass-through is common for funds and changes what investors receive and when.
- **Trader versus investor status.** Several jurisdictions distinguish trading as a business from investing, with different treatment of expenses, losses, and interest. The tests usually involve frequency, holding period, and intent, and they are worth confirming before the first year rather than after.
- **Character and timing of gains.** Short versus long holding periods, mark-to-market elections, and instrument-specific regimes (futures and options are frequently treated differently from equities) can move the effective rate substantially for identical pre-tax returns.
- **Wash sales and loss deferral.** A systematic strategy that re-enters positions quickly can defer losses it believes it has realized. This is a rule that interacts badly with automation, because the code does not know about it.
- **Cross-border.** Withholding on dividends and interest, treaty relief, permanent-establishment risk from where you and your servers sit, and reporting obligations that follow the investor rather than the manager.
- **Employment and carry.** How management fees and performance allocations are characterized for the people receiving them, which is a different question from how the fund is taxed.

The one operational point that is not advice: **tax reporting is a data problem before it is a tax problem.** Lot-level records, corporate-action adjustments, and instrument classifications must exist and be retained from day one, because reconstructing them in year three costs more than maintaining them ever did.

## The operations calendar

Everything above is a schedule. Written on one page, it is also the answer to most of the operational-diligence questionnaire.

| Frequency | Task |
|---|---|
| Pre-open | Confirm data feeds, positions match the broker, risk limits loaded, kill switch reachable |
| Post-close | Reconcile fills, positions, and cash; classify and age every break; back up |
| Daily | Review the P&L against the model's expectation; log any manual intervention |
| Weekly | Clear aged breaks; review capacity and borrow; test the backup restore |
| Monthly | Strike NAV with the administrator; process subscriptions and redemptions; accrue fees and expenses; publish the flash, then the letter |
| Quarterly | Investor call; compliance review; personal-trading attestations; disaster-recovery test |
| Annually | Audit; tax filings; regulatory filings and renewals; insurance; policy review; verify the track record presentation |
| On event | Break escalation, control failure, key-person absence, broker or vendor failure |

The calendar is the deliverable. A manager who can produce it, and yesterday's completed instance of it, has answered the operational due diligence questionnaire more convincingly than any policy document does.

!!! abstract "Key takeaways"
    - Reconciliation is the only control that checks the whole book against an independent record. The test file had **40 fills on both sides and 7 breaks** — every count-based check passed, and only a join on the order identifier found them.
    - The classes are not equally urgent. **Two position breaks totalled 500 shares and $124,346** and make tomorrow's sizing wrong; a **partial fill booked in full** cost 50 shares and diagnosed a bug in fill handling.
    - **Price and fee breaks are trivial money and real information**: $3.00 and $1.50, but the fee break is an exchange charge the cost model does not know about — **$378 a year** and a systematically optimistic backtest.
    - **Timing breaks self-clear at T+1 and are the most numerous**, which is exactly why they must be classified and aged automatically. A timing break still open at T+2 is not a timing break.
    - One stale price on 7% of the book overstated gross assets by **$61,250, or 0.31%** — invisible in a letter, decisive everywhere else.
    - That error moved **$5,048.55 permanently from a subscriber to existing holders**, and crystallized **$12,229.58 of performance fee that was never earned**. The larger error pays the manager, which is the arithmetic argument for an independent administrator.
    - Compliance obligations step at thresholds tied to *whose* money you manage. For quantitative firms, recordkeeping covers research and communications, and performance advertising rules govern exactly the presentation choices measured in the previous lesson.
    - Tax is a data problem first: lot-level records, corporate actions, and classifications must be retained from day one, because reconstructing them later costs more than maintaining them.

## Where this goes next

The calendar above keeps an existing business running. It does not produce the next strategy, and a book that stops adding sleeves has a known trajectory: the edge decays, capacity fills, and the operations run flawlessly on a strategy that no longer works.

[Research Workflow, Hiring, and Scaling](04-research-workflow-hiring-scaling.md) turns the machinery inward, onto the process that generates strategies rather than trades. It measures what a research pipeline without a registry actually produces — an out-of-sample result well below what the in-sample screen promised, from ideas with no edge at all — and then takes up the two things that outlast any single strategy: who you hire, and how a business built around one book becomes one that runs several.
