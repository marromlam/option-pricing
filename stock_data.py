import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class Stock:
    def __init__(self, symbol, sfreq='2m', name=False,
                 start_date='20200101', end_date='20200130',
                 source='es.finance.yahoo.com/',
                 exchange='Unknown',dates=[], price=[],
                 at_open=[], high=[], low=[], at_close=[], volume=[],
                 adj_close=[]):
        self.symbol = symbol
        self.name = name if name else self.symbol
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.exchange = np.copy(exchange)
        self.dates = np.copy(dates)
        self.price = np.copy(price)
        self.at_open = np.copy(at_open)
        self.at_close = np.copy(at_close)
        self.high = np.copy(high)
        self.low = np.copy(low)
        self.volume = np.copy(volume)
        self.adj_close = np.copy(adj_close)
        self.sfreq = sfreq.lower()


    @classmethod
    def from_yahoo(cls, symbol, sfreq='1d',
                   start_date='20200101', end_date='20200130'):
        shell_args = " ".join([symbol, sfreq, start_date, end_date])
        os.system(f"bash yahoo_download.sh {shell_args}")
        try:
            Ta = pd.read_csv(f'{symbol}.csv')
            # Ta = pd.read_csv('AssetsData/REPSOL.csv')
            os.remove(f"{symbol}.csv")
        except:
            raise ValueError('There is no files with this name in dir.')
        c = cls(symbol=symbol,
                sfreq=sfreq,
                source = 'es.finance.yahoo.com/',
                exchange = '',
                dates = pd.to_datetime(Ta['Date']),
                at_open = Ta['Open'],
                high = Ta['High'],
                low = Ta['Low'],
                at_close = Ta['Close'],
                adj_close = Ta['Adj Close'],
                volume = Ta['Volume'],
                price = Ta['Adj Close'],
                )
        return c
        

    def plot(self):
        plt.subplot(4, 1, 1)
        plt.semilogy(self.dates, self.price)
        plt.title(self.symbol)
        plt.ylabel('Adjusted Close')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.grid()

        plt.subplot(4, 1, 3)
        plt.plot(self.dates, self.log_returns)
        plt.title(f'Historical Volatility = {self.volatility:.3f}')
        plt.ylabel('Log Returns')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.grid()

        plt.subplot(4, 1, 4)
        plt.plot(self.dates, self.volume)
        plt.ylabel('Volume')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.grid()
        plt.show()


    @property
    def freq(self):
        # Get mean difference between dates of valid price data
        idx = ~np.isnan(self.price)
        freq = np.mean(np.diff(self.dates[idx]))
        freq = np.timedelta64(freq, 'ms')
        freq = freq.astype(np.float64) * 1e-3 / 3600 / 24
        return freq

    @property
    def period(self):
        # Get maximum date ranges
        idx = ~np.isnan(self.price)
        period = self.dates[idx]
        period = np.max(period) - np.min(period)
        period = np.timedelta64(period, 'ms')
        period = period.astype(np.float64) * 1e-3 / 3600 / 24
        return period
    
    @property
    def n_samples(self):
        return len(self.dates)
    
    @property
    def log_returns(self):
        # Compute Log Returns using Price data
        m, n = np.atleast_2d(self.price).shape
        if n > 1:
            
            return np.concatenate((
                    np.zeros((m,)),
                     np.diff(np.log(self.price))
                     ), axis=0)
        elif n == 1:
            return np.full((1, m), np.nan)
        else:
            return []
    
    @property
    def mean_log_returns(self):
        # Computation of Historical Log Returns. Computes the mean
        # differences in Dates, then annualizes the Log Returns
        return np.mean(self.log_returns) * (365.25 / self.freq)
    
    @property
    def returns(self):
        # Compute Log Returns using Price data
        n, m = self.price.shape
        if n > 1:
            return (np.concatenate((np.zeros((1, m)), np.diff(self.price)), axis=0) / self.price)
        elif n == 1:
            return np.full((1, m), np.nan)
        else:
            return []
    
    @property
    def volatility(self):
        # Estimation of annualized historical volatility. Computes the
        # standard deviation and rescales to an annual basis.
        return self.volatility_yz()
    
    @property
    def volatility_cc(self):
        # Estimation of annualized historical volatility using adjusted
        # close-to-close data. Computes the standard deviation and
        # rescales to an annual basis
        y = self.log_returns
        y = y[~np.isnan(y)]
        return np.std(y) * np.sqrt(365.25 / self.freq)

    def volatility_yz(self):
        n = self.n_samples
        k = 0.34/(1+(n+1)/(n-1))
        sf = self.adj_close/self.at_close
        op = sf*self.at_open
        cl = sf*self.at_close
        so  = np.std(np.log(op[1:]/cl[:-1]), ddof=1)
        sc  = np.std(np.log(self.at_close/self.at_open), ddof=1)
        vrs = np.sum(
                np.log(self.high/self.at_close)*np.log(self.high/self.at_open) + 
                np.log(self.low/self.at_close)*np.log(self.low/self.at_open)
              )/n
        return np.sqrt(so**2 + k*sc**2 + (1-k)*vrs)*np.sqrt(365.25/self.freq)

    # def get_correlations(self):
    #     num_stocks = len(q)
    #     Cor = np.ones((num_stocks, num_stocks))
    #     for i in range(num_stocks - 1):
    #         for j in range(i + 1, num_stocks):
    #             I, J = np.intersect1d(q[i].dates, q[j].dates, return_indices=True)
    #             A = q[i].AdjClose[I]
    #             B = q[j].AdjClose[J]
    #             A = np.log(A[:-1] / A[1:])
    #             B = np.log(B[:-1] / B[1:])
    #             R = np.corrcoef(A, B)
    #             N = len(I)
    #             Cor[i][j] = R[1][0] * N / (N - 1) #unbiassed correction
    #             Cor[j][i] = Cor[i][j]
    #     return Cor
    
    def dumps(self):
        ans = {}
        kans = ['name', 'symbol', 'name', 'start_date', 'end_date', 'source',
                'exchange', 'dates', 'price', 'at_open', 'at_close', 'high',
                'low', 'volume', 'adj_close', 'sfreq','freq', 'volatility']
        for k in kans:
            ans[k] = self.__getattribute__(k)
        return ans

    def __str__(self):
        ans = []
        ans.append(80*'=')             
        ans.append(' {:<17} ({}:{})'.format(self.name, self.exchange, self.symbol))
        ans.append(80*'-')             
        ans.append(f" {'Price History:':<19} {self.dates.min()} to {self.dates.max()}")
        ans.append(f" {'Samples:':<19} {self.n_samples:6d}  samples")
        ans.append(f" {'Period:':<19} {self.period:6.2f}  days")
        ans.append(f" {'Sample rate:':<19} {365.25 / (self.period / self.n_samples):6.2f}")
        ans.append(f" {'Volatility:':<19} {100*self.volatility:6.2f}  % annualized")
        ans.append(f" {'Mean Log Return:':<19} {100*self.mean_log_returns:6.2f}  % annualized")
        ans.append(80*'=')             
        return '\n'.join(ans)


if __name__ == "__main__":
    test = Stock.from_yahoo('AAPL', '1d', '20220121', '20230120')
    vol = test.dumps()['volatility']
    if vol - 0.33713 < 1e-4:
        print(test)
        exit(0)
    else:
        exit(1)
