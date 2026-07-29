# Bayesian Updating

### Continuous $\Theta$, Discrete $K$

coin with bias $\Theta$; prior $f_{\Theta}(\cdot)$

$$f_{\Theta\lvert K}=\dfrac{f_{\Theta}(\theta)\,p_{K\lvert\Theta}(k\lvert\theta)}{p_K(k)}$$

$$p_K(k)=\int f_{\Theta}(\theta')\,p_{K\lvert\Theta}(k\lvert\theta')\,\mathrm{d}\theta'$$

## The Beta Distribution

$$\theta^k(1-\theta)^{n-k}$$

## MAP Estimate

## LMS Estimate

!!! note
    $$\int_{0}^{1}\theta^\alpha(1-\theta)^\beta\mathrm{d}\theta=\dfrac{\alpha!\,\beta!}{(\alpha+\beta+1)!}\qquad\text{for}\quad\alpha,\beta\geq0$$
