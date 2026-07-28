# Trend Following vs Mean Reversion

Put two profitable systematic traders in a room and there is a decent chance they are, at this exact moment, on opposite sides of the same market. One bought crude oil because it has rallied for six weeks; the other sold it because it is three standard deviations above its recent mean. Both can be running sound strategies. Both can make money over the year. This is not a paradox — it is the central duality of systematic trading, and understanding it precisely is worth more than any indicator you will ever plot.

Almost every directional strategy you will encounter is, underneath its branding, a bet on one of two propositions: *moves continue* (trend following, momentum) or *moves reverse* (mean reversion). These are not styles or preferences. They are opposite claims about a measurable statistical quantity, and they produce return streams with opposite shapes.

## Autocorrelation is the signature

Strip away the terminology and the two families reduce to one statistic: the autocorrelation of returns at your trading horizon.

$$\rho_1 = \operatorname{corr}(r_t,\ r_{t+1})$$

If $\rho_1 > 0$ at your horizon, an up period tends to be followed by another up period: the market trends, and buying strength is the correct response. If $\rho_1 < 0$, an up period tends to be followed by a down period: the market mean-reverts, and fading strength is correct. If $\rho \approx 0$ — which is the base case for liquid markets at most horizons — neither family has an edge, and any backtest that says otherwise is fitting noise.

This framing does two useful things. It turns a philosophical debate ("do markets trend?") into an empirical question you can test on data, with confidence intervals. And it makes clear why the two families are mutually exclusive *at a given horizon*: the autocorrelation is either positive, negative, or indistinguishable from zero. You cannot be rewarded for both bets on the same series at the same timescale.

## Convex versus concave: the shape of the P&L

The deeper difference between the families is not their average return — it is the *shape* of their return distribution. This shape follows mechanically from the trade logic, before any market data enters the picture.

A trend follower enters on strength and exits on a stop or a trend break. Most breakouts fail, so most trades are small losses. Occasionally a move runs for months, and the system rides nearly all of it. The result: a low win rate, capped losses, uncapped wins — a **convex** profile with positive skew. It resembles owning options: you pay steady small premiums for occasional large payoffs.

A mean-reversion trader sells strength and buys weakness, targeting a return to fair value. Usually the reversion happens quickly: many small wins, high win rate. But when the "overextended" move is actually new information — a regime change, a merger, a default — the position moves violently against a trader whose logic says to add. The result: a high win rate, capped wins, occasionally severe losses — a **concave** profile with negative skew. It resembles selling options: steady premium income punctuated by rare large payouts.

| Property | Trend following | Mean reversion |
|---|---|---|
| Statistical bet | Positive return autocorrelation | Negative return autocorrelation |
| Win rate | Low (often 30–40% of trades) | High (often 60–80% of trades) |
| Typical win vs typical loss | Rare large wins, many small losses | Many small wins, rare large losses |
| Return skew | Positive | Negative |
| Analogy | Long options: pay premium, own the tail | Short options: collect premium, sell the tail |
| P&L rhythm | Long flat/bleeding stretches, occasional surges | Smooth equity curve, sudden air pockets |
| Characteristic failure | Whipsaw: choppy, trendless markets | Regime break: the spread never comes back |
| Crisis behavior | Often profitable (rides the panic) | Often the source of the blowup |

!!! warning "Sharpe ratio hides the shape"
    Two strategies with identical Sharpe ratios can have opposite skew. The concave strategy's risk is concentrated in rare events that a few years of backtest may simply not contain — its measured Sharpe flatters it. When you evaluate any strategy, ask *where the risk lives*: spread evenly through time, or hiding in the tail?

## Horizon dependence: the same market does both

Here is the fact that dissolves most "do markets trend or revert?" arguments: the answer depends on the measurement horizon, and a single market routinely does both at once.

Broad empirical regularities documented across decades of academic and practitioner research follow a rough pattern in equities: at horizons of minutes to a few days, returns tend weakly toward reversal (liquidity provision earns the spread); at horizons of roughly 3–12 months, relative returns show momentum — winners keep winning; at horizons of 3–5 years, they show reversal again — long-run winners underperform. Different economic mechanisms operate at each timescale: inventory and order-flow effects at the short end, underreaction and flow-chasing in the middle, overreaction and mean-reverting valuations at the long end.

The practical consequence is that "AAPL is trending" is an incomplete sentence. Trending *at what horizon*? A stock can be in a 6-month uptrend (momentum long), 3 standard deviations above its 5-day mean (short-term reversion short), and historically expensive (long-horizon reversal short) simultaneously — and three well-built systems can hold three different positions in it, each with a legitimate edge at its own timescale.

!!! example "Design implication"
    A strategy's holding period must match the horizon at which its statistical signature actually exists. A common retail failure mode is testing a momentum signal at a horizon where the data mean-reverts (or vice versa), finding nothing, and concluding "momentum doesn't work" — when the real error was a horizon mismatch.

## The archetypes, and why each is psychologically brutal

Both families have mature institutional embodiments, and their histories are instructive without needing to cite any particular fund's numbers.

The trend archetype is the **managed-futures CTA**: systematic programs trading diversified futures — rates, FX, commodities, equity indices — across dozens of markets, following medium-term trends with strict volatility-scaled sizing. The industry's structural appeal is its crisis behavior: because major crises are themselves large sustained trends (down in equities, up in bonds and volatility), trend programs have historically tended to perform well precisely when conventional portfolios suffer — the "crisis alpha" argument.

The reversion archetype is the **equity statistical-arbitrage desk**: market-neutral books holding hundreds or thousands of long and short positions, betting that relative mispricings between similar stocks correct within days or weeks. Individually noisy bets become a smooth aggregate through sheer breadth — the law of large numbers as a business model.

Each demands a different kind of discipline, and the difficulty is exactly dual:

- **Trend following punishes you daily and rewards you rarely.** You will lose on most trades, endure drawdowns lasting a year or more, and watch open profits evaporate at every trend's end — because giving back the last 20% of each move is the price of catching the rare 300% move. The temptation is to "improve" the system by taking profits early, which surgically removes the convexity that is the entire edge.
- **Mean reversion rewards you daily and punishes you rarely — and enormously.** Months of steady gains build false confidence; then a position moves against you and the system's own logic says the trade just got *better*. Distinguishing "more attractive entry" from "the world changed" is the hardest judgment in the family, and averaging into a genuine regime break is how reversion traders die. Hard risk limits exist because in the moment, the addictive smoothness of the equity curve argues against them.

Neither difficulty is a design flaw to engineer away. Each is the *cost* of the corresponding payoff shape — remove it and you remove the edge.

!!! abstract "Key takeaways"
    - Trend following and mean reversion are opposite bets on one statistic: the sign of return autocorrelation at the trading horizon.
    - Trend following is convex — low win rate, small losses, rare huge wins, positive skew, option-buyer economics.
    - Mean reversion is concave — high win rate, small wins, rare severe losses, negative skew, option-seller economics.
    - Sharpe ratios conceal skew; always ask whether a strategy's risk is spread through time or concentrated in the tail.
    - Autocorrelation is horizon-dependent: the same market can revert intraday, trend over months, and revert again over years — strategy horizon must match signal horizon.
    - The psychological cost is dual: trend followers endure constant small losses waiting for outliers; reversion traders endure rare catastrophes interrupting constant small wins. Both costs are inseparable from the edge.

## Where this goes next

The mean-reversion family's institutional form deserves its own treatment: how do you bet on reversion without betting on market direction at all? The answer — trade the *spread* between related instruments rather than the instruments themselves — is the foundation of an entire industry, and it is where we go next: [Statistical Arbitrage](08-statistical-arbitrage.md).
