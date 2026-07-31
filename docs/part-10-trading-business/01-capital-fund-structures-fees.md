# Capital, Fund Structures, and Fees

[Profiling, Refactoring, and Versioning](../part-09-software-engineering/05-profiling-refactoring-versioning.md) closed Part IX with a platform that can be trusted: reproducible from a commit hash, gated by CI, profiled rather than guessed at. What it cannot tell you is whose money should run through it. That question has three answers — your own, a firm's, or an investor's — and they differ less in the trading than in everything around it.

This lesson measures two things that are usually only asserted. The first is fee path dependence, and the measured result is not the one the argument is usually about: with an annual high-water mark, the **performance fee is almost perfectly path-independent — 1.5% spread across three paths with identical gross returns** — while the **management fee spreads by 34%**. The second is break-even AUM, where an intraday strategy turns out to need **28% of its entire capacity** deployed before the founder clears $300,000, against **$1.3 million of own capital** for the same take-home.

!!! note "Scope"
    Everything here is an orientation to how these structures work, not legal, tax, or investment advice; the fee terms, cost lines, and thresholds are illustrative and vary by jurisdiction, vehicle, and negotiation. Both blocks are self-contained — no market data, no external services — so the numbers reproduce anywhere.

## Whose money is at risk decides everything else

The structure question looks like a question about fees. It is really a question about who absorbs the first loss, because that determines control, obligations, and how much of the upside is yours to keep.

| | Own capital | Prop firm | First-loss | Fund (LP/GP) |
|---|---|---|---|---|
| First loss absorbed by | You | The firm | You, to a stated buffer | The investors |
| Typical upside to you | 100% | 30–50% | 70–90% | 2% of AUM + 20% of profit |
| Capital you must post | All of it | None | 10–20% of the line | Usually a GP commitment |
| Who controls the risk limits | You | The firm, unilaterally | The firm, against your buffer | You, within the offering documents |
| Reporting obligations | Tax filing | Internal | Internal | Audited NAV, investor letters, regulator |
| Time to start | Days | Weeks | Weeks | Six to twelve months |
| Annual fixed cost | Data and execution | None | Platform fee | See the break-even block below |

The two middle columns are where most people underestimate the terms. A prop seat pays no management fee and posts no capital, which sounds strictly better until the firm halves your line after a bad month — a decision you do not participate in. A first-loss arrangement inverts that: you post the buffer, so you keep most of the profit, but you have bought a leveraged position in your own strategy and the platform fee is charged whether you trade well or not.

A managed account sits outside this table because it is a distribution channel rather than a structure: the investor's assets stay in the investor's name at the investor's broker, you trade them under a limited power of attorney, and you get fee revenue without a fund wrapper. It is the cheapest way to run outside capital and the reason many managers never form a fund at all. The cost is that every account is separately administered and separately reconciled, which scales badly past a handful of them.

## The high-water mark works; the management fee is what moves

The standard worry about performance fees is that they are path-dependent — that a manager who loses and recovers gets paid for the same dollar twice. The high-water mark exists to prevent exactly that. Whether it does is measurable: take one set of five annual returns, reorder it, and add a third path with the same compound return and no volatility at all.

```python
import numpy as np

NAV0 = 10_000_000.0
MGMT, PERF = 0.02, 0.20

# The same five annual gross returns in two different orders, plus a smooth
# path with the identical five-year compound return and no volatility at all.
sawtooth = [0.30, -0.20, 0.30, -0.20, 0.40]
total = float(np.prod([1 + r for r in sawtooth]))
paths = {
    "sawtooth": sawtooth,
    "recovery": [-0.20, -0.20, 0.30, 0.30, 0.40],
    "smooth": [total ** 0.2 - 1] * 5,
}


def run(rets, hurdle=0.0):
    """Annual crystallization: 2% on beginning NAV, 20% above the high-water mark."""
    nav, hwm, mgmt_paid, perf_paid = NAV0, NAV0, 0.0, 0.0
    for r in rets:
        mgmt = MGMT * nav
        nav = nav * (1 + r) - mgmt
        perf = PERF * max(nav - hwm * (1 + hurdle), 0.0)
        nav -= perf
        hwm = max(hwm, nav)
        mgmt_paid, perf_paid = mgmt_paid + mgmt, perf_paid + perf
    return mgmt_paid, perf_paid, nav


print(f"  every path ends at the same gross NAV: {NAV0 * total:,.0f} ({total - 1:+.2%})")
print(f"  {'path':9s} {'mgmt fee':>11s} {'perf fee':>11s} {'to manager':>11s} "
      f"{'investor':>12s} {'hurdle gain':>12s}")
for name, rets in paths.items():
    mgmt, perf, nav = run(rets)
    _, _, nav_h = run(rets, hurdle=0.05)
    print(f"  {name:9s} {mgmt:11,.0f} {perf:11,.0f} {mgmt + perf:11,.0f} "
          f"{nav:12,.0f} {nav_h - nav:12,.0f}")
# =>   every path ends at the same gross NAV: 15,142,400 (+51.42%)
#      path         mgmt fee    perf fee  to manager     investor  hurdle gain
#      sawtooth    1,070,791     742,811   1,813,602   12,971,245      229,374
#      recovery      832,791     751,175   1,583,966   13,004,700      100,000
#      smooth      1,112,254     739,905   1,852,159   12,959,620      627,033
```

The performance fee is the boring column. Across a sawtooth, a two-year drawdown followed by recovery, and a straight line, it comes to **742,811, 751,175, and 739,905 — a spread of 1.5%** on paths whose year-to-year returns have nothing in common. The high-water mark does its job almost exactly: unpaid recovery cancels the extra volatility, and what the manager is paid for is the compound return, which is identical by construction.

The management fee is where the path actually shows up. It ranges from **832,791 to 1,112,254, a spread of 34%**, for the mechanical reason that it is charged on NAV and NAV is larger for longer on a path that gains early. Total take to the manager therefore spans **1,583,966 to 1,852,159 — $268,193, or 17%** — and essentially all of that difference is the flat fee nobody argues about. Note also the sign: the manager does *worst* on the path that loses first and best on the smooth one, so the fee structure is not, as is often claimed, a reward for volatility.

The hurdle column is the one that should change a negotiation. A 5% hurdle returns **627,033 to the investor on the smooth path and 100,000 on the recovery path — a factor of 6.3**. A hurdle is usually pitched as protection against a manager who is being paid for beta or for luck; measured, it bites hardest on the steady, modest manager whose annual return sits just above it, and barely touches the volatile one whose good years clear any hurdle by a mile.

One number is in none of the columns. The investor ends the sawtooth path at 12,971,245 against a gross 15,142,400, giving up **2,171,155 to hand the manager 1,813,602**. The missing $357,553 is not a fee — it is the return the investor never earned on fees already paid out. The investor always loses more than the manager gains, and the gap grows with the holding period.

## The management fee is a salary; the performance fee is an option

Those two columns behave differently because they are different instruments, and the incentives follow from the payoff shapes rather than from anybody's character.

The management fee is a claim on assets. It is paid in bad years, it does not depend on returns, and it scales with the one variable the manager controls directly — how much money is raised. A manager living on management fees is running an asset-gathering business, and the failure mode is well documented: capacity is quietly exceeded because turning money away is expensive and the degradation shows up slowly.

The performance fee is a call option on the fund's return, struck at the high-water mark, and the manager cannot be short it. Options are worth more when volatility is higher, which is the standard argument that performance fees encourage risk-taking. The block above shows why that argument is weaker than it sounds in the annual-crystallization case — the high-water mark reclaims the volatility premium — but it becomes correct at exactly the moment it matters most: a manager far *below* the high-water mark holds a deeply out-of-the-money option, and the only way to bring it back into the money is to take more risk. Deferred crystallization, a claw-back, or a real GP commitment are the structural answers; none of them are rhetorical.

## Break-even AUM is a capacity question, not a fundraising question

The fund wrapper is not free, and its cost is close to fixed — an audit costs roughly the same on $10 million as on $100 million. That fixed cost sets a floor on viable AUM, and the floor should be compared not to what you hope to raise but to what the strategy can actually absorb.

```python
MGMT, PERF, TAKE_HOME = 0.02, 0.20, 300_000.0

fund_costs = {                     # annual, a small fund with one employee
    "market data and research feeds": 60_000,
    "execution, hosting, connectivity": 45_000,
    "fund admin, audit, tax preparation": 95_000,
    "legal and compliance": 70_000,
    "one non-founder salary, fully loaded": 180_000,
}
solo_costs = 25_000                # data and execution only; no fund wrapper
FIXED = sum(fund_costs.values())

strategies = [                     # name, expected gross return, capacity
    ("intraday stat-arb", 0.25, 40e6),
    ("cross-sectional equity", 0.12, 300e6),
    ("multi-asset trend", 0.08, 2000e6),
]

for k, v in fund_costs.items():
    print(f"  {k:38s} {v:>9,}")
print(f"  {'annual fixed cost of the fund wrapper':38s} {FIXED:>9,}")
print()
print(f"  {'strategy':23s} {'fee rate':>9s} {'break-even':>11s} {'+300k':>11s} "
      f"{'% capacity':>11s} {'own capital':>12s}")
for name, gross, cap in strategies:
    rate = MGMT + PERF * max(gross - MGMT, 0.0)          # manager revenue per $1 of AUM
    breakeven = FIXED / rate
    target = (FIXED + TAKE_HOME) / rate
    solo = (TAKE_HOME + solo_costs) / gross              # your own money, no fees, no wrapper
    print(f"  {name:23s} {rate:9.2%} {breakeven / 1e6:10.1f}M {target / 1e6:10.1f}M "
          f"{target / cap:11.1%} {solo / 1e6:11.1f}M")
# =>   market data and research feeds            60,000
#      execution, hosting, connectivity          45,000
#      fund admin, audit, tax preparation        95,000
#      legal and compliance                      70,000
#      one non-founder salary, fully loaded     180,000
#      annual fixed cost of the fund wrapper    450,000
#
#      strategy                 fee rate  break-even       +300k  % capacity  own capital
#      intraday stat-arb           6.60%        6.8M       11.4M       28.4%         1.3M
#      cross-sectional equity      4.00%       11.2M       18.8M        6.2%         2.7M
#      multi-asset trend           3.20%       14.1M       23.4M        1.2%         4.1M
```

Read the break-even column first and it looks backwards. The intraday strategy breaks even at **$6.8 million** and the trend book needs **$14.1 million**, because a 25% gross return earns the manager 6.60% of AUM against the trend book's 3.20%. High returns lower the AUM you need. That is the intuitive result and it is the less important one.

The capacity column is the one that decides. Paying the founder $300,000 on top of costs takes **28.4% of the intraday strategy's entire $40 million capacity**, against 6.2% and 1.2% for the other two. The high-return strategy is the *harder* business, because its edge lives in a small pool and the fixed costs of the wrapper consume a real fraction of it — and it will hit capacity, degrade, and stop clearing the same fee rate long before the trend book notices it has investors. Capacity, not expected return, is what makes a strategy fundable.

The last column reframes the whole exercise. The same $300,000 take-home needs **$1.3 million to $4.1 million of your own capital** and no wrapper at all: no audit, no administrator, no compliance counsel, no employee, no investors. Choosing the fund path is a bet that you can raise between five and ten times more money than you could ever accumulate yourself — which is often true, and is a fundraising and operations problem rather than a trading one.

## Choosing a path for the first five years

The decision is mostly determined by two facts you already know: how much capital you can put at risk, and how long a live, independently verifiable track record you have.

| Own capital | Live track record | Reasonable path |
|---|---|---|
| Under $250k | None | Trade your own account. A prop or first-loss seat is worth taking only if the platform fee is small relative to the line, and read the drawdown clause before the profit split. |
| $250k–$2M | Under two years | Own account, and start the record properly now: separately custodied, monthly, net of everything. The record is the asset you are building. |
| $250k–$2M | Two years or more | Managed accounts for friendly capital. Fee revenue with no wrapper cost, and it proves the operations before an allocator tests them. |
| Over $2M | Three years or more | A fund, but only if the strategy's capacity clears the break-even by an order of magnitude, and only with an anchor investor already identified. |

The common mistake is forming the fund first. The wrapper costs money from the day it exists, the audited track record it produces does not start until it is funded, and there is no allocator on earth who will look at a fund with a two-month record. The cheaper sequence is to build the record in an account you already control, then let the structure follow the capital that wants in.

!!! abstract "Key takeaways"
    - The structure question is really about who absorbs the first loss. Prop capital costs you control — a firm can halve your line unilaterally — and first-loss arrangements make you a leveraged buyer of your own strategy.
    - With an annual high-water mark, the performance fee is nearly path-independent: **742,811 / 751,175 / 739,905 across three paths with identical compound returns, a spread of 1.5%**. The mark does what it claims to.
    - The path dependence lives in the management fee, which spread **34%** (832,791 to 1,112,254) because it is charged on NAV. Total take to the manager ranged **$268,193, or 17%**, and the manager did worst on the path that lost first.
    - A 5% hurdle was worth **627,033 to the smooth investor and 100,000 to the drawdown investor — 6.3× more against the steady manager** than the volatile one, the reverse of how hurdles are usually pitched.
    - The investor gives up more than the manager receives: **2,171,155 surrendered to pay 1,813,602**, the $357,553 gap being return never earned on fees already paid.
    - A fund wrapper costs about **$450,000 a year** and is nearly fixed in AUM. Paying the founder $300,000 over that costs **28.4% of an intraday strategy's capacity** but **1.2% of a trend book's** — capacity, not expected return, is what makes a strategy fundable.
    - The same take-home needs **$1.3M–$4.1M of your own capital**. The fund path is a bet on raising five to ten times what you could save, and it is an operations bet, not a trading one.

## Where this goes next

Every number above assumed the investor shows up. They do not show up for a fee schedule; they show up for a track record they believe and an operation they have inspected. The break-even AUM in this lesson is the amount you must raise, and raising it means putting a monthly report in front of someone whose job is to find the reason to say no.

[Investor Reporting and Due Diligence](02-investor-reporting-due-diligence.md) builds that report, then attacks it. The same underlying track is passed through four presentation choices — each individually defensible, each one an allocator has seen before — and the reported Sharpe roughly doubles without a single false statement being made.
