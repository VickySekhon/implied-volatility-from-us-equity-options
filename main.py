import yfinance as yf
from datetime import date
import math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import griddata

print("PICTON Investments Case Study By: Vicky Sekhon")

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
        time_to_expiry: float,
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
        ) - strike_price * math.e ** (
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
        time_to_expiry: float,
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
        time_to_expiry: float,
        tol=1e-10,
        max_iterations=100,
    ):
        # Begin guessing at 20% volatility
        sigma = 0.2
        for _ in range(max_iterations):
            predicted_price = self.black_scholes(
                sigma, stock_price, strike_price, time_to_expiry
            )
            error = predicted_price - actual_price

            if abs(error) < tol:
                return sigma, predicted_price

            vega = self._vega(sigma, stock_price, strike_price, time_to_expiry)
            print(vega)

            if vega < tol:
                print(
                    f"Vega is too small to continue, implied volatility was not found."
                )
                return None

            sigma = sigma - error / vega      
            sigma = max(1e-6, min(sigma, 10.0))  # clamp between ~0% and 1000% vol

        print(f"Reached end of maximum iterations, implied volatility was not found.")
        return None

class Options_Data:
    def __init__(self):
        self.yf_obj = yf
        self.ticker_df = None
        self.ticker_history_df = None
        self.options = None

    def load_data(self, tickers: str, expiry_date=None) -> None:
        ticker_df = self.yf_obj.Ticker(tickers)
        
        if expiry_date:
          option_df = ticker_df.option_chain(date=expiry_date)
        else:
          option_df = ticker_df.option_chain()
          
        option_df = ticker_df.option_chain(date=expiry_date)
        self._set_ticker_df(ticker_df)
        self._set_option_df(option_df)

    def download_option_data(self, ticker: str) -> None:
        assert self.options is not None, "Cannot download data, load it first."

        destination_dir = os.path.join(os.curdir, "data")
        file_name = os.path.join(destination_dir, f"{ticker}_option_data.csv")

        os.makedirs(destination_dir, exist_ok=True)
        self.options.to_csv(file_name)

    # This returns ticker data not option data
    def download_historical_data(self, tickers: str, **kwargs) -> None:
        history_df = self.yf_obj.download(tickers, **kwargs)
        self._set_ticker_history(history_df)

        destination_dir = os.path.join(os.curdir, "data")
        file_name = os.path.join(destination_dir, f"ticker_data.csv")

        os.makedirs(destination_dir, exist_ok=True)
        history_df.to_csv(file_name)

    def _set_option_df(self, option_df):
        self.options = option_df

    def _set_ticker_df(self, ticker_df):
        self.ticker_df = ticker_df

    def _set_ticker_history(self, history_df):
        self.ticker_history_df = history_df
        
        
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

def main():
    data = Options_Data()
    IV = Implied_Volatility()

    expiry = date(2026, 3, 27)
    today = date.today()
    time_to_expiry = (expiry - today).days / 365

    data.load_data("AAPL")
    stock_price = data.ticker_df.history(period="1d")["Close"].iloc[-1]


    plot = {"strike": [], "implied_volatility": [], "time_to_expiry": []}

    for expiry_str in data.ticker_df.options:
        expiry = date.fromisoformat(expiry_str)
        time_to_expiry = (expiry - today).days / 365

        if time_to_expiry <= 0:
            continue

        chain = data.ticker_df.option_chain(expiry_str)
        calls = chain.calls
        puts = chain.puts
        
        option_chain_data = pd.concat([calls, puts], ignore_index=True)
        
        option_chain_data.fillna(0, inplace=True)

        for _, row in option_chain_data.iterrows():
            if row["bid"] == 0 or row["ask"] == 0:
                continue  # skip illiquid

            actual_price = (row["bid"] + row["ask"]) / 2
            strike_price = row["strike"]
            
            lower = stock_price * 0.80   # within 20% below spot
            upper = stock_price * 1.05   # within 20% above spot

            if strike_price < lower or strike_price > upper:
                continue

            if row["volume"] < 10:
                continue

            if actual_price < 0.05:
                continue
              
            if row["openInterest"] < 100:
                continue
              
            if time_to_expiry < (14/365):
                continue

            iv = IV.newton_raphson(actual_price, stock_price, strike_price, time_to_expiry)

            if iv is None:
                continue
              
            iv, predicted_price = iv

            print(f"actual price {actual_price}, stock price {stock_price}, strike price {strike_price}, time to expiry {time_to_expiry}, iv {iv}, predicted price {predicted_price}")
            plot["strike"].append(strike_price)
            plot["time_to_expiry"].append(time_to_expiry)
            plot["implied_volatility"].append(iv)


    strikes = np.array(plot["strike"])
    times = np.array(plot["time_to_expiry"])
    ivs = np.array(plot["implied_volatility"])


    strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
    time_grid = np.linspace(times.min(), times.max(), 50)
    S_grid, T_grid = np.meshgrid(strike_grid, time_grid)

  
    IV_grid = griddata((strikes, times), ivs, (S_grid, T_grid), method="linear")
    # IV_grid = np.clip(IV_grid, 0.3, 0.45)


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, T_grid, IV_grid, cmap="RdYlGn_r")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to Expiry (Years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("AAPL Implied Volatility Surface")
    plt.show()
    return
  
if __name__ == "__main__":
    main()