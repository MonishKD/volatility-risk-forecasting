"""Volatility forecasting research code (QUARCC).

Timing convention used everywhere
---------------------------------
Sessions are indexed by trading date t. The log return r[t] = ln(P[t] / P[t-1]) is known only
after the close of session t. A forecast with origin t is issued after that close, uses
information through r[t] only, and predicts variance of the future sessions t+1 .. t+h.
All quantities are in daily variance units (squared log returns), never standard deviations.
"""
