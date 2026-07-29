# Law of Total Variance

The variance of $X$ decomposes into the expected within-group variance and the variance of the group means:

$$\mathrm{var}(X)=\mathbb{E}[\mathrm{var}(X\lvert Y)]+\mathrm{var}\!\left(\mathbb{E}[X\lvert Y]\right)$$

## Sum of a Random Number of Independent Random Variables

If $Y=X_1+\ldots+X_N$ where the $X_i$ are i.i.d. and $N$ is itself a random variable independent of the $X_i$,

$$\mathrm{var}(Y)=\mathbb{E}[N]\,\mathrm{var}(X)+(\mathbb{E}[X])^2\,\mathrm{var}(N)$$

!!! warning "Draft"
    This page is a partial draft — the full treatment is planned but not yet written.
