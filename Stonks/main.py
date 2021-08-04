import numpy as np
import random

from QuantConnect import Resolution
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Indicators import *

ALL_UNIVERSE = True
VOL_CUTOFF = 5000000
CBI_CUTOFF = 0.5
VOL_TREND_DAYS = 7
RSI_TREND_DAYS = 7
MACD_DIV_TREND_DAYS = 4
STOCH_SLOW_TREND_DAYS = 4
RSI_LOW = 40.0
RSI_HI = 75.0
D_SLOW_LOW = 10.0
D_SLOW_HI = 50.0
EACH_TRADE_AMT = 10000

class Stonks(QCAlgorithm):
    def Initialize(self):
            self.SetStartDate(2015, 1, 1)
            self.SetEndDate(2018, 12, 31)
            self.InitCash = 100*EACH_TRADE_AMT
            self.SetCash(self.InitCash)
            self.UniverseSettings.Resolution = Resolution.Daily
            self.indicators = { }
            if ALL_UNIVERSE:
                self.AddUniverse(self.CoarseSelectionFunction)
            else:
                self.AddEquity("A", Resolution.Daily)
                self.indicators["A"] = SelectionData()
            
            self.AddEquity("SPY", Resolution.Daily)

            self.SetBenchmark("SPY")
            
            self.MKT = self.AddEquity("SPY", Resolution.Daily).Symbol
            self.mkt = []
            self.order_tracker = {}
        
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
        self.Log(f"OnSecuritiesChanged({self.Time}):: {changes}")

            
    def OnData(self, slice):
        self.Debug("On data")
        # Update the indicators
        for symbol, sd in self.indicators.items():
            if slice.Bars.ContainsKey(symbol):
                bar = slice.Bars[symbol]
                sd.Update(bar)
                self.Log(f"Time: {bar.EndTime} {sd.Print()}")

        if self.IsWarmingUp:
            return
        
        if not self.Portfolio.Invested:
            self.Debug("Buying SPY")
            self.SetHoldings("SPY", 1)
        
        self.Log(f"All keys is {slice.Keys}")
        for symbol in slice.Keys:
            if symbol == "SPY":
                continue
            sd = self.indicators[symbol]
            if sd.should_buy():
                self.Debug(f"Buying {symbol} at price {self.Securities[symbol].Close}")
                # Sell some amount of SPY
                spy_qty = min(int(EACH_TRADE_AMT/self.Securities["SPY"].Close), self.Portfolio["SPY"].Quantity)
                market_ticket = self.MarketOrder("SPY", -1*spy_qty, False, f"Selling SPY for {symbol}")
                self.order_tracker[market_ticket.OrderId] = symbol
            # If invested, and if sell signal is true
            elif self.Portfolio[symbol].Invested and sd.should_sell():
                self.Debug(f"Selling {symbol}")
                self.Liquidate(symbol)
                 # Buy some amount of SPY
                spy_qty = int(self.Portfolio.Cash/self.Securities["SPY"].Close)
                if spy_qty >= 1:
                    self.MarketOrder("SPY", spy_qty, False, f"Buying SPY after selling {symbol}")
            else:
                self.Log(f"Neither buy or sell {symbol}")
                
    def OnEndOfDay(self, symbol):
        if symbol != self.MKT:
            return
        mkt_price = self.History(self.MKT, 2, Resolution.Daily)['close'].unstack(level= 0).iloc[-1]
        self.mkt.append(mkt_price)
        mkt_perf = self.InitCash * self.mkt[-1] / self.mkt[0] 
        self.Plot('Strategy Equity', self.MKT, mkt_perf)
        
    def OnOrderEvent(self, orderEvent):
        order = self.Transactions.GetOrderById(orderEvent.OrderId)
        if orderEvent.Status == OrderStatus.Filled:
            self.Log("{0}: {1}: {2}".format(self.Time, order, orderEvent))
            if orderEvent.OrderId not in self.order_tracker.keys():
                return
            symbol = self.order_tracker.pop(orderEvent.OrderId)
            sd = self.indicators[symbol]
            sym_qty = int(min(EACH_TRADE_AMT, self.Portfolio.Cash)/(self.Securities[symbol].Close * 1.2))
            self.Debug(f"To sell {sym_qty}")
            self.MarketOrder(symbol, sym_qty)
            self.StopMarketOrder(symbol, -1 * sym_qty, sd.sma_200.Current.Value)
            
class SelectionData():
    def __init__(self):
        self.stoch = Stochastic(14, 3, 3)
        self.rsi = RelativeStrengthIndex(14)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.Exponential)
        self.d_slow_rw = RollingWindow[float](STOCH_SLOW_TREND_DAYS)
        self.rsi_rw = RollingWindow[float](RSI_TREND_DAYS)
        self.macd_div_rw = RollingWindow[float](MACD_DIV_TREND_DAYS)
        self.vol_rw = RollingWindow[float](VOL_TREND_DAYS)
        self.sma_10 = SimpleMovingAverage(10)
        self.sma_20 = SimpleMovingAverage(20)
        self.sma_50 = SimpleMovingAverage(50)
        self.sma_200 = SimpleMovingAverage(200)
        self.current_price = 0.0
    
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
        self.sma_10.Update(bar.EndTime, bar.Close)
        self.sma_20.Update(bar.EndTime, bar.Close)
        self.sma_50.Update(bar.EndTime, bar.Close)
        self.sma_200.Update(bar.EndTime, bar.Close)
        self.current_price = bar.Close
        
    def Print(self):
        return f"Stoch Fast and slowKD: {self.stoch.Current.Value} {self.stoch.StochK} {self.stoch.StochD}\
        MACD: {self.macd.Current.Value} RSI: {self.rsi.Current.Value}"
        
    def should_buy(self):
        CBI_CUTOFF = 1.0
        macd_div = self.macd.Current.Value - self.macd.Signal.Current.Value
        filtered = self.Avg_Volume > VOL_CUTOFF and macd_div < 0 and self.MACD_Div_slope > 0 and self.MACD_Div_int < 0 \
            and self.rsi.Current.Value > RSI_LOW and self.rsi.Current.Value < RSI_HI and self.RSI_slope > 0 \
            and self.current_price > self.sma_20.Current.Value and self.sma_50.Current.Value > self.sma_200.Current.Value
        if not filtered:
            return False
        else:
            return (0.5*((abs(self.stoch.StochD.Current.Value - 0.5*(D_SLOW_HI + D_SLOW_LOW))/(0.5*(D_SLOW_HI - D_SLOW_LOW))) \
                + (macd_div/self.MACD_Div_int)) > CBI_CUTOFF)

    def should_sell(self):
        return (self.sma_10.Current.Value < self.sma_50.Current.Value)
        # self.MACD_Div_slope <= 0 and self.MACD_Div_int >= 0 \
        # and self.stoch.StochD.Current.Value >= 50 and self.RSI_slope <= 0 
