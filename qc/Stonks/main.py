import numpy as np
import random

from QuantConnect import Resolution
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Indicators import *

ALL_UNIVERSE = True
VOL_CUTOFF = 2000000
CBI_CUTOFF = 0.5
VOL_TREND_DAYS = 7
RSI_TREND_DAYS = 7
MACD_DIV_TREND_DAYS = 4
STOCH_SLOW_TREND_DAYS = 4
RSI_LOW = 40.0
RSI_HI = 75.0
D_SLOW_LOW = 10.0
D_SLOW_HI = 50.0

class Stonks(QCAlgorithm):
    def Initialize(self):
            self.SetStartDate(2019, 3, 1)
            self.SetEndDate(2019, 5, 1)
            self.SetCash(100000)
            self.UniverseSettings.Resolution = Resolution.Daily
            self.indicators = { }
            if ALL_UNIVERSE:
                self.AddUniverse(self.CoarseSelectionFunction)
            else:
                self.AddEquity("A", Resolution.Daily)
                self.indicators["A"] = SelectionData()
            
            self.SetBenchmark("SPY")
            # Warm-up the indicator with bar count
            self.SetWarmUp(28, Resolution.Daily)

        
    def CoarseSelectionFunction(self, universe):
        selection = []
        universe = sorted(universe, key=lambda c: c.DollarVolume, reverse=True)  
        universe = [c for c in universe if c.Price > 10][:1000]
        
        for coarse in universe:
            symbol = coarse.Symbol
            if symbol not in self.indicators:
                self.indicators[symbol] = SelectionData()
            selection.append(symbol)

        return selection

    def OnSecuritiesChanged(self, changes):
        self.Debug(f"OnSecuritiesChanged({self.Time}):: {changes}")

            
    def OnData(self, slice):
        self.Debug("On data")
        # Update the indicators
        for symbol, sd in self.indicators.items():
            if slice.Bars.ContainsKey(symbol):
                bar = slice.Bars[symbol]
                sd.Update(bar)
                self.Debug(f"Time: {bar.EndTime} {sd.Print()}")

        if self.IsWarmingUp:
            return
        
        self.Log(f"All keys is {slice.Keys}")
        for symbol in slice.Keys:
            self.Debug(f"Key is {symbol}")
            sd = self.indicators[symbol]
            if sd.should_buy():
                self.Debug(f"Buying {symbol}")
                self.MarketOrder(symbol, 1)
                self.StopMarketOrder(symbol, -1, 0.95 * self.Securities[symbol].Close)
            # If invested, and if sell signal is true, and if current price is more than avg buying price
            elif self.Portfolio[symbol].Invested and sd.should_sell() \
                and self.Securities[symbol].Close > self.Portfolio[symbol].AveragePrice:
                self.Debug(f"Selling {symbol}")
                self.Liquidate(symbol)
                # self.MarketOrder(symbol, -1*self.Portfolio[symbol].Holdings.Quantity)
            else:
                self.Debug(f"Neither buy or sell {symbol}")
            
            
class SelectionData():
    def __init__(self):
        self.stoch = Stochastic(14, 3, 3)
        self.rsi = RelativeStrengthIndex(14)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.Exponential)
        self.d_slow_rw = RollingWindow[float](STOCH_SLOW_TREND_DAYS)
        self.rsi_rw = RollingWindow[float](RSI_TREND_DAYS)
        self.macd_div_rw = RollingWindow[float](MACD_DIV_TREND_DAYS)
        self.vol_rw = RollingWindow[float](VOL_TREND_DAYS)
    
    def calc_slope(self, rw):
        x = list(rw)[::-1]
        x_len = len(x)
        if x_len < 2:
            return None
        slope = np.polyfit(range(len(x)), x, 1)[0]
        return slope

    def calc_intercept(self, rw):
        x = list(rw)[::-1]
        x_len = len(x)
        if x_len < 2:
            return None
        intercept = np.polyfit(range(len(x)), x, 1)[1]
        return intercept
    
    def calc_mean(self, rw):
        return np.mean(list(rw))
    
    @property
    def is_ready(self):
        return self.stoch.IsReady and self.rsi.IsReady and self.macd.IsReady
        
    @property
    def RSI_slope(self):
        return self.calc_slope(self.rsi_rw)
        
    @property
    def d_slow_slope(self):
        return self.calc_slope(self.d_slow_rw)
        
    @property
    def MACD_Div_slope(self):
        return self.calc_slope(self.macd_div_rw)
        
    @property
    def MACD_Div_int(self):
        return self.calc_intercept(self.macd_div_rw)

    @property
    def Avg_Volume(self):
        return self.calc_mean(self.vol_rw)
        
    def Update(self, bar):
        self.stoch.Update(bar)
        self.rsi.Update(bar.EndTime, bar.Close)
        self.macd.Update(bar.EndTime, bar.Close)
        self.rsi_rw.Add(self.rsi.Current.Value)
        self.macd_div_rw.Add(self.macd.Current.Value - self.macd.Signal.Current.Value)
        self.d_slow_rw.Add(self.stoch.StochD.Current.Value)
        self.vol_rw.Add(bar.Volume)
        
    def Print(self):
        return f"Stoch Fast and slowKD: {self.stoch.Current.Value} {self.stoch.StochK} {self.stoch.StochD}\
        MACD: {self.macd.Current.Value} RSI: {self.rsi.Current.Value}"
        
    def should_buy(self):
        CBI_CUTOFF = 1.0
        macd_div = self.macd.Current.Value - self.macd.Signal.Current.Value
        filtered = self.Avg_Volume > VOL_CUTOFF and self.rsi.Current.Value > RSI_LOW and self.rsi.Current.Value < RSI_HI \
            and self.RSI_slope > 0 and macd_div < 0 and self.MACD_Div_slope > 0 and self.MACD_Div_int < 0
        if not filtered:
            return False
        else:
            return 0.5*((abs(self.stoch.StochD.Current.Value - 0.5*(D_SLOW_HI + D_SLOW_LOW))/(0.5*(D_SLOW_HI - D_SLOW_LOW))) \
                + (macd_div/self.MACD_Div_int)) > CBI_CUTOFF

    def should_sell(self):
        return self.RSI_slope <= 0 and self.MACD_Div_slope <= 0 and self.MACD_Div_int >= 0 \
            and self.stoch.StochD.Current.Value >= 50
