---
layout: post
title: "What Is Your Personal Inflation?"
date: 2026-10-08
description: "Official inflation measures the price change of an average consumer basket. But how closely does that basket resemble yours?"
categories: [economics, data, inflation, projects]
---

A few days ago, I was discussing inflation with a group of friends when I realized that none of them had a clear idea of how the official rate is actually calculated.

We hear about inflation constantly but that single number hides an important assumption: it represents an *average* consumer.

But none of us is exactly average.

## What does inflation measure?

Inflation is a broad increase in the prices of goods and services over time. When prices rise faster than our income, the purchasing power of money falls: the same amount of money buys fewer things than before.

Measuring this is more complicated than checking whether a few products have become more expensive. Prices vary across products, shops, regions and periods. Some prices rise while others remain stable or even fall.

In Italy, consumer-price indices are calculated by [ISTAT](https://www.istat.it/), the national statistical institute. ISTAT collects millions of price observations and organizes goods and services into a hierarchical classification known as ECOICOP.

The next step is crucial: each category receives a **weight** based on its importance in the average household budget.

An increase in the price of bread matters more to the overall index than an equal increase in the price of a product that almost nobody buys. If the price of moon rocks suddenly doubled, the official cost of living would be largely unaffected. If the price of food, electricity or rent increased, the effect would be much greater.

Inflation is therefore not a simple average of price changes. It is a **weighted average**.

## The official basket is not your basket

The weighting system makes the official index meaningful for the population as a whole. However, it cannot perfectly represent every individual.

For example, the 2026 Italian NIC basket assigns approximately 2.9% of total expenditure to alcoholic beverages, tobacco and narcotics. That may be a reasonable population-level estimate, but it clearly overstates this category for someone who never buys those products. At the same time, it may understate it for an alcoholic.

This does not mean that official inflation is inaccurate. It means that it answers a different question:

> How much did prices change for the average national consumer basket?

Personal inflation instead asks:

> How much did prices change for a basket that resembles my own spending?

## Estimating personal inflation

To explore this difference, I developed the [Personal Inflation Calculator](https://mypersonalinflation.streamlit.app/).

The application allows users to enter their spending as annual amounts or percentage shares. Expenses can be assigned to broad categories—such as housing, transport and food—or to more detailed subcategories.

The calculation follows three steps:

1. Each expense is converted into a share of total expenditure.
2. Those personal weights are applied to the corresponding ISTAT price indices.
3. The annual change in the resulting personal index is compared with the official NIC inflation rate.

The application also identifies the categories in which the personal basket differs most from the official ISTAT basket. This helps explain *why* the two inflation rates diverge.

## My result

I tested the calculator using my own spending distribution and obtained the following result for 2025:

![My personal inflation compared with the official ISTAT rate](/assets/images/my-inflation-plot.png)

My estimated personal inflation was **2.35%**, while the official ISTAT rate was **1.65%**. My result was therefore **0.70 percentage points higher** than the official figure.

That difference does not mean that every product I bought became 0.70% more expensive. It means that the categories carrying more weight in my budget experienced, in combination, stronger price growth than the categories emphasized by the official basket.

The historical chart also shows that this relationship changes over time. In some years my basket would have produced higher inflation; in others, slightly lower inflation. There is no permanent rule saying that an individual must always experience more or less inflation than the national average.

## What the estimate cannot tell us

A personal inflation calculator is still a simplified model.

It assumes that the spending distribution entered today remains constant throughout the historical period. In reality, people adapt: they may change supermarket, use less energy, replace a product or stop buying something that has become too expensive.

The estimate also depends on the level of detail of the available categories. Two people may both spend on transport but buy very different products within that category.

For these reasons, the result should not be interpreted as an exact measurement of every price a person faced. Its value is explanatory: it shows how strongly the composition of a basket influences the inflation rate attached to it.

---

## Sources and links

- [Try the Personal Inflation Calculator](https://mypersonalinflation.streamlit.app/)
- [Project repository](https://github.com/Tommaso-Vigano/personal-inflation)
- [European Central Bank — What is inflation?](https://www.ecb.europa.eu/ecb-and-you/explainers/tell-me-more/html/what_is_inflation.en.html)
- [ISTAT — Consumer price indices, basket and weighting updates for 2026](https://www.istat.it/en/press-release/consumer-price-indices-year-2026/)

