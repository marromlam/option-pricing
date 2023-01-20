## Repsol & Iberdrola [R;I]

from stock_data import Stock


stock1 = Stock.from_yahoo('AAPL', '1d', '20220121', '20230120')
stock1.plot()
print(stock1)
print(stock1.dumps())
