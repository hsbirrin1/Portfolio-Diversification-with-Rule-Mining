# Portfolio-Diversification-with-Rule-Mining

Quant Portfolio

A Python project that combines fundamental analysis, quantitative scoring, and association rule mining to evaluate companies and build a data-driven portfolio allocation strategy.

Features

* SEC Data Extraction
Pulls financial statement data directly from the SEC EDGAR API.

* Financial Ratios
Return on Equity (ROE)
Return on Assets (ROA)
Debt-to-Equity (D/E, inverted for scoring)
Net Margin
Return on Invested Capital (ROIC)
Free Cash Flow Margin
EPS Growth (5 years)

* Health Score (0–1 scale)
Normalizes ratios via min–max scaling across peers.
Averages metrics into a single HealthScore for each company.
Higher = stronger fundamentals relative to peers.

* Association Rule Mining (Apriori)
Uses MLxtend’s Apriori algorithm to find hidden patterns.
Companies with Positive EPS Growth often have Medium HealthScore.

* Portfolio Allocation
Allocates a fixed investment amount ($100k) across companies.
Weighting tilted toward firms with higher HealthScores.
Generates a pie chart of the allocation.

* Outputs
Raw ratios table
Scaled ratios + HealthScores
Apriori rules discovered
Final portfolio allocation

* Tech Stack
Python (3.10+)
Pandas / NumPy → data wrangling
Requests → SEC API calls
MLxtend → Apriori association rule mining
Matplotlib → portfolio pie chart visualization

This project also includes an event study around the WHO COVID-19 pandemic declaration (2020-03-11) for Alphabet (GOOGL). The purpose of the event test was to evaluate how companies with strong fundamentals, as measured by the HealthScore, perform under market stress. By analyzing the market reaction during the COVID-19 shock, we could see whether high-scoring companies, like Alphabet (GOOGL), were more resilient and able to recover faster. This step ensured that the portfolio was not only built on financial strength but also tested for durability in real-world crisis conditions.
