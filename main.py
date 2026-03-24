import yfinance as yf
import math, os
import numpy as np
import pandas as pd
from scipy.stats import norm

print("PICTON Investments Case Study By: Vicky Sekhon")


"""
goal is given prices of options, determine the implied volatility (future volatility)

black scholes takes in volatility and prices an option

numerical root-finding
for an option, we can use black schloes to guess the volatility by repeatedly subbing in a random volatility until the options price is output
- plug in vol -> output is too low, guess higher vol 
- plug in vol -> output is too high, guess lower vol 

Volatility surface is a map of fear: Z = fear (IV), X = when expiry date is X, Y = where strike price is Y
volatility surface trends:
- Vol skew: OTM puts are bought a lot since they have a low premium and hedge managers use them to protect against potentially huge crashes
- the red is near-term, OTM puts, which are extremely inexpensive
- Blue trough: ATM short-term calls have a higher premium and a lower IV because there is less uncertainty about them expiring OTM or near it
- jagged red IV that smooths out over time shows that short-dated options are more sensitive to fear than long-dated ones
- price of underlying = ATM = vol minimum = blue trough in this graph

the more volatile an option is, the more incentivized buyers are to purchase it since it could be profitable. If it isn't, then the max a buyer loses is their option premium. 

price of underlying moves more -> higher volatility -> higher option premium -> higher implied volatility

IV is what the market expects moving forward (market is going to move -> higher volatility moving forward -> options are priced higher)

The volatility surface is a function of strike, K, and time-to-maturity, 
T, and is defined implicitly as: C(S, K, T) := BS (S, T, r, q, K, σ(K, T))    
 
C(S, K, T) denotes the current market price of a call option with time-to-maturity T and strike K

"""


class Implied_Volatility:
    THREE_MONTH_T_BILL_RATE = 0.037149999999999996

    def __init__(self):
        return

    """ 
    Step 1: Calculate d1 and d2
    Where:

    S = current stock price
    K = strike price
    r = risk-free interest rate
    σ = volatility (this is what you're guessing when finding IV)
    T = time to expiry (in years)
    """

    def black_scholes(
        self,
        volatility: float,
        stock_price: float,
        strike_price: float,
        time_to_expiry: int,
    ) -> float:
        d1 = (
            math.log((stock_price / strike_price))
            + (self.THREE_MONTH_T_BILL_RATE + ((volatility**2) / 2)) * time_to_expiry
        ) / (volatility * (math.sqrt(time_to_expiry)))

        # Probability the option expires ITM
        d2 = d1 - (volatility * (math.sqrt(time_to_expiry)))

        # Call price = what you expect to gain - what you pay
        call_price = stock_price * self._probability_from_normal_distribution(
            d1
        ) - strike_price * math.e**(
            -self.THREE_MONTH_T_BILL_RATE * time_to_expiry
        ) * self._probability_from_normal_distribution(
            d2
        )
        return call_price

    def _probability_from_normal_distribution(self, d1: float) -> float:
        return norm.cdf(d1)

    # Derivative of BS price w.r.t volatility
    # Tells us the change in price given some volatility
    # vega = 0.20 tells us that +1% change in volatility = $0.20 increase in price
    def _vega(
        self,
        volatility: float,
        stock_price: float,
        strike_price: float,
        time_to_expiry: int,
    ) -> float:
        d1 = (
            math.log((stock_price / strike_price))
            + (self.THREE_MONTH_T_BILL_RATE + ((volatility**2) / 2)) * time_to_expiry
        ) / (volatility * (math.sqrt(time_to_expiry)))

        return stock_price * (math.sqrt(time_to_expiry)) * norm.pdf(d1)

    # Numerical root-finding = newton-raphson
    def newton_raphson(
        self,
        actual_price: float,
        stock_price: float,
        strike_price: float,
        time_to_expiry: int,
        tol=1e-6,
        max_iterations=100,
    ):
        # Begin guessing at 20% volatility
        sigma = 0.2
        for _ in range(max_iterations):
            predicted_price = self.black_scholes(
                sigma, stock_price, strike_price, time_to_expiry
            )
            error = actual_price - predicted_price

            if abs(error) < tol:
                return sigma

            vega = self._vega(sigma, stock_price, strike_price, time_to_expiry)

            if vega < tol:
                print(
                    f"Vega is too small to continue, implied volatility was not found."
                )
                return None

            sigma = sigma - error / vega

        print(f"Reached end of maximum iterations, implied volatility was not found.")
        return None
      
      
IV = Implied_Volatility()
# Generate a fake market price first
S, K, T = 4500, 4500, 1.0
fake_market_price = IV.black_scholes(0.2, S, K, T)

# Now recover it
iv = IV.newton_raphson(fake_market_price, S, K, T)
print(f"Original market price: ${fake_market_price:.2f} calculated at 0.2 volatility.")
print(f"Recovered IV: {iv}")



class Options_Data:
    def __init__(self):
        self.yf_obj = yf
        self.df = None
        
    def download_data(self, tickers: list[str], **kwargs) -> None:
        df = self.yf_obj.download(tickers, **kwargs)
        self._set_data(df)
        
        destination_dir = os.path.join(os.curdir, "data")
        file_name = os.path.join(destination_dir, f"options_data.csv")
        
        os.makedirs(destination_dir, exist_ok=True)
        df.to_csv(file_name)
        
    def _set_data(self, df):
        self.df = df
        
aapl = yf.Ticker("aapl")  
print(aapl)
# aapl_history = aapl.history(period="5d", interval="1h")
# print(len(aapl_history))

#data = yf.download()

data = Options_Data()
data.download_data("AAPL", period="5d", interval="1h")
print(data.df.head())
print(data.df.tail())