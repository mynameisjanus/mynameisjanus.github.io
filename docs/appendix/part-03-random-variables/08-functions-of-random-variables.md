# Functions of Random Variables

## Linear Function of a Discrete Random Variable

$$Y=aX+b$$

$$p_Y(y)=p_X\!\left(\frac{y-b}{a}\right)$$

## Linear Function of a Continuous Random Variable

$$Y=aX+b$$

$$f_Y(y)=\dfrac{1}{\lvert a\rvert}f_X\!\left(\frac{y-b}{a}\right)$$

## General Function $g(X)$ of a Continuous Random Variable

Two steps:
* Find the CDF of $Y$
* Differentiate

## Functions of Multiple Random Variables

Now, consider a function of two random variables $g(X,Y)$. It is also a random variable, i.e., $g(X,Y)=Z$ with PMF

$$p_Z(z)=\mathbf{P}(g(X,Y)=z)=\sum_{(x,y):\,g(x,y)=z}p_{X,Y}(x,y).$$

The latter states that we have to sum the joint probability mass function over all pairs $(x,y)$ such that $g(x,y)=z$. Its expectation is

$$\mathbb{E}[g(X,Y)]=\sum_{x}\sum_{y}g(x,y)p_{X,Y}(x,y).$$
