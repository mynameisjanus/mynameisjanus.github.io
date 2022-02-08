---
title: Introduction
sidebar: probability_sidebar
permalink: probability_intro.html
folder: probability
toc: false
usemathjax: true
---

<p align="center">
  <img src="images/prob/red-dice.png" style="width:100px;height:auto;"/>
</p>

<br>

Uncertainty is a fundamental aspect of nature. Let's say we want to roll a die. Before we roll it or before it lands, we don't know what the outcome of the experiment will be. When we invest our money in the market, we are uncertain whether the value of our investments will go up or down. Even in areas of our lives where we might believe the certainty of particular outcomes, there will be uncertainty, no matter how infinitesimal. The mathematical language for describing and quantifying uncertainty is **probability**.

## Sample Space

For a given experiment, we will associate a list or set of possible outcomes. This is called a **sample space** and we denote this by $$\Omega$$. In the example of rolling a die, the sample space is the set $$\{1,2,3,4,5,6\}$$. A sample space can be ***discrete*** or ***continuous***. It can also be ***finite*** or ***infinite***.

For a list or set to be a valid sample space, it must be:

* mutually exclusive,
* collectively exhaustive, and
* at the *right* granularity.

We now consider rolling a die twice. We can list all the possible outcomes of this experiment through a sequential description. First, we know that the possible results of the first roll are $$\{1,2,3,4,5,6\}$$. For the second roll, we have the same set. To get the sample space, we simply pair each element from the first roll with each element from the second roll. That is, the sample space for rolling a die twice is $$\Omega=\{(1,1),(1,2),\ldots,(2,1),(2,2),\ldots,(6,5),(6,6)\}$$.

## Event

An **event** is simply a *subset* of the sample space. In the example of rolling a die, *getting an even number* is an event. We denote an event by a capital letter, say $$A$$, and we will often speak of the ***probability of an event*** $$A$$.
