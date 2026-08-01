# Multinomial Distribution

This is the first law in this part that describes several counts at once, and the interesting content is not in any one of them. Each individual count is a binomial and says nothing new. What is new is that the counts are negatively correlated — not approximately, not as an empirical regularity, but by construction, because they are forced to sum to $n$. That constraint is the entire multivariate content of the family, and it is the reason a table of category shares cannot be read one row at a time.

This page covers the coefficient that counts arrangements with repetition, the joint mass function, the binomial margins, the covariance the constraint induces, and the consequence for reading a table of estimated shares. It does not cover the two-category case, which is [Binomial Distribution](02-binomial-distribution.md); it does not cover the same experiment without replacement; it does not build the goodness-of-fit statistic that summarises all the cells at once, which is [Chi-Square Distribution](15-chi-square-distribution.md); and it treats no covariance matrix as an object in its own right, which is [Part VI](../part-06-multivariate-probability/index.md).

The trading stake is a regime table. Classify twenty-five years of days into four states and report each state's share with a standard error beside it, and every one of those standard errors will be right while the table as a whole is misleading. The last section quantifies by how much: four intervals each covering $95\%$ individually cover all four simultaneously only $83\%$ of the time, and the negative correlation the constraint imposes is what stops that number falling further.

## The Counting Coefficient

Run $n$ independent trials, each landing in one of $r$ categories with probabilities $p_1,\ldots,p_r$ summing to one, and let $N_j$ count the trials landing in category $j$. The vector $(N_1,\ldots,N_r)$ is multinomial. Its mass function needs a count of the sequences producing a given tally, and that count is the multinomial coefficient

$$\binom{n}{n_1,\ldots,n_r}=\frac{n!}{n_1!\,n_2!\cdots n_r!}.$$

With $r=2$ this is the ordinary binomial coefficient, since $n!/(k!(n-k)!)$ is the same object written with the second index left implicit.

??? note "Proof that the coefficient counts arrangements of a word with repeated letters"
    Order the $n$ trials in a row and write each one's category as a letter. A tally $(n_1,\ldots,n_r)$ corresponds to a word using letter $j$ exactly $n_j$ times, and the number of distinct such words is what we want.

    Suppose first that all $n$ letters were distinguishable — tag them with subscripts. There are $n!$ arrangements. Now remove the tags. Any two arrangements that differ only by permuting the $n_1$ copies of letter $1$ among themselves were counted separately and should not have been, and the same holds independently for each other letter. So each distinct word was counted $n_1!\,n_2!\cdots n_r!$ times, and dividing gives the coefficient.

    The argument is the division principle of [Counting Principles](../part-01-mathematical-foundations/02-counting-principles.md) applied $r$ times, and the load-bearing step is that the overcounting factor is the *same* for every word — which holds because the tally is fixed. It fails immediately if the categories are allowed to have different tallies across the arrangements being counted, which is why the coefficient appears inside a sum over tallies rather than outside it.

## The Joint Mass Function

Every sequence with tally $(n_1,\ldots,n_r)$ has probability $p_1^{n_1}\cdots p_r^{n_r}$ by independence, and the exponents record only how many of each category appeared. Multiplying by the number of such sequences,

$$\mathbf{P}(N_1=n_1,\ldots,N_r=n_r)=\frac{n!}{n_1!\cdots n_r!}\,p_1^{n_1}\cdots p_r^{n_r},\qquad \sum_j n_j=n.$$

The support is a lattice simplex rather than a box: the counts are not free to vary independently, because fixing $r-1$ of them fixes the last. So although the vector has $r$ components it has only $r-1$ degrees of freedom, and the same is true of the probability vector. That deficiency of exactly one is the thread running through the rest of the page, and it reappears as the degrees of freedom of the goodness-of-fit statistic.

## Every Margin Is Binomial

Ask about one category and ignore the rest. Each trial either lands in category $j$ or does not, the trials are independent, and the probability is $p_j$ every time — so

$$N_j\sim\mathrm{Binom}(n,p_j),\qquad \mathbb{E}[N_j]=np_j,\qquad \mathrm{var}(N_j)=np_j(1-p_j).$$

Collapsing the other $r-1$ categories into a single "everything else" is legitimate precisely because the trials are independent and the categories are exhaustive, and it is the reason the margins carry no new information. Anything computed from one row of a multinomial table is a binomial calculation, with all the properties [Binomial Distribution](02-binomial-distribution.md) established, including the overdispersion that appears when the trials are actually dependent.

## The Counts Are Negatively Correlated by Construction

The joint structure is where the family earns its own page:

$$\mathrm{cov}(N_i,N_j)=-np_ip_j\quad(i\ne j),\qquad \mathrm{corr}(N_i,N_j)=-\sqrt{\frac{p_ip_j}{(1-p_i)(1-p_j)}}.$$

??? note "Proof that the covariance is -n p_i p_j, and that the constraint is what forces it"
    Write $N_i=\sum_{t=1}^{n}A_t$ and $N_j=\sum_{t=1}^{n}B_t$, where $A_t$ and $B_t$ indicate that trial $t$ landed in category $i$ and $j$ respectively. Trials are independent, so $\mathrm{cov}(A_s,B_t)=0$ for $s\ne t$ and only the $n$ diagonal terms survive. Within a single trial the two indicators are mutually exclusive — one trial cannot land in two categories — so $A_tB_t=0$ identically and

    $$\mathrm{cov}(A_t,B_t)=\mathbb{E}[A_tB_t]-\mathbb{E}[A_t]\mathbb{E}[B_t]=0-p_ip_j=-p_ip_j.$$

    Summing over $t$ gives $-np_ip_j$. Dividing by the product of the standard deviations $\sqrt{np_i(1-p_i)}\sqrt{np_j(1-p_j)}$ cancels the $n$ entirely, which is the striking part: the *correlation* between two category counts does not depend on the sample size at all, so it cannot be reduced by collecting more data.

    The mutual exclusivity within a trial is the load-bearing hypothesis, and it is worth seeing that it is the summation constraint in local form. A trial that could be assigned to several categories at once — an overlapping classification, a day belonging to two regimes — would have $\mathbb{E}[A_tB_t]>0$, the covariance could take either sign, and none of what follows would hold. The negative sign here is not a fact about markets; it is arithmetic imposed by a partition.

```python
import numpy as np

rng = np.random.default_rng(43)
p = np.array([0.10, 0.25, 0.45, 0.20])                         # four regimes, 25 years of days
n, reps = 6410, 200_000
x = rng.multinomial(n, p, reps)
print(f"  Multinomial(n = {n}, p = {p})")
print("      j     mean      exact       var      exact     margin is Binom?")
for j in range(4):
    print(f"    {j:3d} {x[:, j].mean():9.1f} {n * p[j]:10.1f}"
          f" {x[:, j].var():10.1f} {n * p[j] * (1 - p[j]):10.1f}"
          f"      {abs(x[:, j].var() / (n * p[j] * (1 - p[j])) - 1) < 0.02}")
c = np.cov(x.T)
print("      i  j     corr      exact    sample size cancels?")
for i, j in ((0, 1), (0, 3), (2, 3)):
    exact = -np.sqrt(p[i] * p[j] / ((1 - p[i]) * (1 - p[j])))
    print(f"    {i:3d} {j:2d} {c[i, j] / np.sqrt(c[i, i] * c[j, j]):9.4f} {exact:10.4f}"
          f"       n does not appear")
# =>   Multinomial(n = 6410, p = [0.1  0.25 0.45 0.2 ])
#          j     mean      exact       var      exact     margin is Binom?
#          0     641.0      641.0      578.3      576.9      True
#          1    1602.5     1602.5     1204.4     1201.9      True
#          2    2884.4     2884.5     1585.7     1586.5      True
#          3    1282.0     1282.0     1026.1     1025.6      True
#          i  j     corr      exact    sample size cancels?
#          0  1   -0.1939    -0.1925       n does not appear
#          0  3   -0.1663    -0.1667       n does not appear
#          2  3   -0.4518    -0.4523       n does not appear
```

The first table confirms that nothing multivariate is visible from a single column: every margin has the binomial mean and the binomial variance, to within Monte Carlo noise. The second table is what the margins cannot show. The correlations are all negative, they match the closed form, and — as the proof promised — the expression producing them contains no $n$. Six thousand days of history give exactly the same correlation between two regime counts as sixty would.

!!! note "The correlation between two category counts is a property of the partition, not of the sample"
    Almost every other quantity in this part sharpens as $n$ grows. This one does not move at all, because both the covariance and the two standard deviations scale as $n$ and the ratio cancels. The practical consequence is that the dependence between cells of a contingency table can never be sampled away, and any procedure that treats the cells as independent is making an error that more data will not fix — it will merely make the error's consequences more precisely wrong.

## Why Category Shares Cannot Be Tested One at a Time

Now the trading stake. A regime table reports four shares, each with a standard error computed from the binomial margin, and each interval is individually correct. The question is what the table says jointly.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(47)
p = np.array([0.10, 0.25, 0.45, 0.20])
n, reps = 6410, 200_000
x = rng.multinomial(n, p, reps) / n
se = np.sqrt(p * (1 - p) / n)
inside = np.abs(x - p) <= 1.96 * se                            # each interval, individually
print(f"  four 95% intervals on the shares of {n} days")
for j in range(4):
    print(f"    regime {j}: p {p[j]:.2f}   se {se[j]:.5f}"
          f"   individual coverage {inside[:, j].mean():.4f}")
print(f"  all four simultaneously             {inside.all(axis=1).mean():.4f}")
print(f"  were the four counts independent    {np.prod(inside.mean(axis=0)):.4f}")
worst = (np.abs(x - p) / se).max(axis=1)                       # the widest miss in each table
print(f"  z needed for 95% joint coverage:  exact {np.quantile(worst, 0.95):.4f}"
      f"   Bonferroni {norm.ppf(1 - 0.05 / (2 * len(p))):.4f}")
# =>   four 95% intervals on the shares of 6410 days
#        regime 0: p 0.10   se 0.00375   individual coverage 0.9512
#        regime 1: p 0.25   se 0.00541   individual coverage 0.9498
#        regime 2: p 0.45   se 0.00621   individual coverage 0.9496
#        regime 3: p 0.20   se 0.00500   individual coverage 0.9479
#      all four simultaneously             0.8318
#      were the four counts independent    0.8133
#      z needed for 95% joint coverage:  exact 2.4668   Bonferroni 2.4977
```

Each interval covers its own share $95\%$ of the time, exactly as advertised. All four hold simultaneously only $83.2\%$ of the time — so one row in six of such tables contains at least one interval that has missed, and a reader checking whether the regime mix has shifted will be wrong far more often than the $5\%$ the individual figures imply.

The negative correlation is doing something helpful here, and it is worth noticing which direction. Four genuinely independent intervals would give joint coverage of $0.95^4\approx0.81$; the constraint pushes the observed figure up rather than down, because a share that comes in high forces some other share to come in low, and the errors partially offset instead of accumulating. So the dependence is not an extra problem stacked on the multiplicity problem — it is a mild correction in the safe direction, sitting on top of a multiplicity problem that is real and much larger. The last line prices the repair: widening every interval to $z=2.47$ restores joint coverage to $95\%$, and a Bonferroni correction asks for $2.50$, which is close enough that the crude fix costs almost nothing here.

!!! warning "A table of shares with individually correct standard errors is not a table anybody can read row by row"
    The failure has two independent sources and they are easy to conflate. One is multiplicity, which is [Part XV](../part-15-multiple-testing/index.md) and applies to any table of $r$ simultaneous claims. The other is the constraint, which is specific to a partition and means the cells are not separate measurements at all — with $r-1$ degrees of freedom among $r$ reported numbers, the last row is a deterministic function of the others and carries no independent evidence whatever. The correct object is a single statistic on the whole vector, which is what a goodness-of-fit test computes and why its degrees of freedom are $r-1$ rather than $r$; the distribution of that statistic is [Chi-Square Distribution](15-chi-square-distribution.md).

So the multinomial is best understood as the law that makes a partition into a random vector, and everything distinctive about it traces to the single linear constraint that defines a partition. The margins are old news, the coefficient is combinatorics from Part I, and the one genuinely new object is a covariance whose sign is fixed and whose magnitude no amount of data will change. The practical rule is that whenever a report presents shares of a whole, the shares are not $r$ measurements — they are $r-1$, plus a number that was determined before it was computed.
