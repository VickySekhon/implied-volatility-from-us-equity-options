# Implied-volatility-from-us-equity-options

Python tool that constructs implied volatility surfaces from publicly available U.S. equity options data

# Intuition

The goal is, given some option data, to determine the implied volatility so that a volatility surface can be constructed.

`Black Scholes` is a formula that takes in some volatility level and prices an option. It can be reverse-engineered to continuously guess the volatility while running `Black Scholes` until we reach our actual premium. The converging volatility is determined using `vega` which is the derivative of an option's price with respect to some volatility level and plugging it into the `Newton Raphson` formula to iteratively find our best approximation. Once our output matches our option premium, the correct `implied volatility` is determined.

# Volatility Surface Trends

The volatility surface is essentially an indicator of the markets fear given some option contract comprised of an expiry date and strike price.

Common trends include:

- `Skew` which is concerned with `implied volatility` and `strike price`
  - `Smirk`: OTM puts trade at higher implied volatility than OTM calls
  - `Smile`: Forex options usually explode up and down with equal likelihood
- `Term structure`: which is concerned with `implied volatility` and `time to expiry`
  - `Contago`: short-term implied volatility is low while long-term implied volatility is high
  - `Backwardation`: short-term implied volatility is high, long-term implied volatility is low (good time to sell options rather than buy)

# Author

Vicky Sekhon
