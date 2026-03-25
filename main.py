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
    THREE_MONTH_T_BILL_RATE = 3.63 / 100

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
                return sigma

            vega = self._vega(sigma, stock_price, strike_price, time_to_expiry)

            if vega < tol:
                return None

            sigma = sigma - error / vega
            # Clamp volatility so it is non-negative
            sigma = max(1e-6, min(sigma, 10.0))

        print(f"Reached end of maximum iterations, implied volatility was not found.")
        return None


class Options_Data:
    def __init__(self):
        self.yf_obj = yf
        self.ticker_df = None
        self.ticker_history_df = None
        self.options = None

    def load_data(self, ticker: str, expiry_date=None) -> None:
        ticker_df = self.yf_obj.Ticker(ticker)

        if expiry_date:
            option_df = ticker_df.option_chain(date=expiry_date)
        else:
            option_df = ticker_df.option_chain()

        self._set_ticker_df(ticker_df)
        self._set_option_df(option_df)

    def download_option_data(self, ticker: str) -> None:
        assert self.options is not None, "Cannot download data, load it first."

        destination_dir = os.path.join(os.curdir, "data")
        file_name = os.path.join(destination_dir, f"{ticker}_option_data.csv")

        os.makedirs(destination_dir, exist_ok=True)
        self.options.to_csv(file_name)

    # This returns ticker data not option data
    def download_historical_data(self, ticker: str, **kwargs) -> None:
        history_df = self.yf_obj.download(ticker, **kwargs)
        self._set_ticker_history(history_df)

        destination_dir = os.path.join(os.curdir, "data")
        file_name = os.path.join(destination_dir, f"{ticker}_historical_data.csv")

        os.makedirs(destination_dir, exist_ok=True)
        history_df.to_csv(file_name)

    def get_stock_price(self):
        return self.ticker_df.history(period="1d")["Close"].iloc[-1]

    def get_option_expiries(self):
        return self.ticker_df.options

    def get_option_chain(self, expiry: str):
        return self.ticker_df.option_chain(expiry)

    def _set_option_df(self, option_df):
        self.options = option_df

    def _set_ticker_df(self, ticker_df):
        self.ticker_df = ticker_df

    def _set_ticker_history(self, history_df):
        self.ticker_history_df = history_df


class Utils:
    UPPER_STRIKE_RANGE = 1.05
    LOWER_STRIKE_RANGE = 0.8

    def __init__(self):
        return

    """
    Filtration pipeline that removes outliers from data before generating 
    volatility surface. The goal is to get the volatility surface to be smooth.
    
    1. Remove illiquid assets
    2. Filter for strikes that are within 20% of current underlying price
    3. Filter options that are not heavily traded since they have an unreliable price 
    4. Filter options with very low premiums
    5. Filter options that nobody is holding
    6. Filter options that are expiring within next 2 weeks since they will have a high implied volatility
    """

    def skip_row(
        self,
        row: pd.Series,
        actual_price: float,
        stock_price: float,
        strike_price: float,
        time_to_expiry: int,
    ) -> bool:
        lower = stock_price * self.LOWER_STRIKE_RANGE
        upper = stock_price * self.UPPER_STRIKE_RANGE

        if row["bid"] <= 0 or row["ask"] <= 0:
            return True

        if strike_price < lower or strike_price > upper:
            return True

        if row["volume"] < 10:
            return True

        if actual_price < 0.05:
            return True

        if row["openInterest"] < 100:
            return True

        if time_to_expiry < (14 / 365):
            return True

        return False

    def get_average_price(self, ask, bid):
        return (ask + bid) / 2

    def plot_3d_surface(self, data_points, ticker):
        strikes = np.array(data_points["strike"])
        times = np.array(data_points["time_to_expiry"])
        ivs = np.array(data_points["implied_volatility"])

        strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
        time_grid = np.linspace(times.min(), times.max(), 50)
        S_grid, T_grid = np.meshgrid(strike_grid, time_grid)

        IV_grid = griddata((strikes, times), ivs, (S_grid, T_grid), method="linear")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(S_grid, T_grid, IV_grid, cmap="viridis")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Time to Expiry")
        ax.set_zlabel("Implied Volatility")
        ax.set_title(f"{ticker} Implied Volatility Surface")
        Utils().save_figure(ticker, plt)

    def save_figure(self, ticker, plt):
        destination_dir = os.path.join(os.curdir, "surfaces")
        file_name = os.path.join(destination_dir, f"{ticker}.png")
        os.makedirs(destination_dir, exist_ok=True)
        plt.savefig(file_name, dpi=150, bbox_inches="tight")


def main(ticker: str):
    data = Options_Data()
    IV = Implied_Volatility()
    utils = Utils()

    data.load_data(ticker)
    stock_price = data.get_stock_price()

    total_values = 0
    total_filtered = 0

    data_points = {"strike": [], "implied_volatility": [], "time_to_expiry": []}

    for expiry_str in data.get_option_expiries():
        expiry = date.fromisoformat(expiry_str)
        time_to_expiry = (expiry - date.today()).days / 365

        if time_to_expiry <= 0:
            continue

        option_chain = data.get_option_chain(expiry_str)
        calls, puts = option_chain.calls, option_chain.puts
        option_chain_data = pd.concat([calls, puts], ignore_index=True)
        option_chain_data.fillna(0, inplace=True)
        total_values += len(option_chain_data)

        for _, row in option_chain_data.iterrows():
            actual_price = utils.get_average_price(row["bid"], row["ask"])
            strike_price = row["strike"]

            skip_row = utils.skip_row(
                row, actual_price, stock_price, strike_price, time_to_expiry
            )
            if skip_row:
                total_filtered += 1
                continue

            iv = IV.newton_raphson(
                actual_price, stock_price, strike_price, time_to_expiry
            )

            if iv is None:
                continue

            data_points["strike"].append(strike_price)
            data_points["time_to_expiry"].append(time_to_expiry)
            data_points["implied_volatility"].append(iv)

    print(
        f"Ticker {ticker} - total values before filtration {total_values}\nTotal values after filtration {total_filtered}"
    )
    utils.plot_3d_surface(data_points, ticker)


if __name__ == "__main__":
    main("AAPL")
    main("GOOG")
    main("VOO")
