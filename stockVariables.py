import pandas as pd
stocks = pd.read_csv('./static/tickers.csv')
stocks['Last Sale'] = stocks['Last Sale'].str.replace('$', '')
stocks['Last Sale'] = pd.to_numeric(stocks['Last Sale'], downcast='float')
type(stocks['Last Sale'][0])

price_filter = stocks['Last Sale'] >= 3.00
cap_filter = stocks['Market Cap'] >= 100000000
stocks = set(stocks[(price_filter) & (cap_filter)]['Symbol'])
blacklist = {'I', 'ELON', 'WSB', 'THE', 'A', 'ROPE', 'YOLO', 'TOS', 'CEO', 'DD', 'IT', 'OPEN', 'ATH', 'PM', 'IRS', 'FOR','DEC', 'BE', 'IMO', 'ALL', 'RH', 'EV', 'TOS', 'CFO', 'CTO', 'DD', 'BTFD', 'WSB', 'OK', 'PDT', 'RH', 'KYS', 'FD', 'TYS', 'US', 'USA', 'IT', 'ATH', 'RIP', 'BMW', 'GDP', 'OTM', 'ATM', 'ITM', 'IMO', 'LOL', 'AM', 'BE', 'PR', 'PRAY', 'PT', 'FBI', 'SEC', 'GOD', 'NOT', 'POS', 'FOMO', 'TL;DR', 'EDIT', 'STILL', 'WTF', 'RAW', 'PM', 'LMAO', 'LMFAO', 'ROFL', 'EZ', 'RED', 'BEZOS', 'TICK', 'IS', 'PM', 'LPT', 'GOAT', 'FL', 'CA', 'IL', 'MACD', 'HQ', 'OP', 'PS', 'AH', 'TL', 'JAN', 'FEB', 'JUL', 'AUG', 'SEP', 'SEPT', 'OCT', 'NOV', 'FDA', 'IV', 'ER', 'IPO', 'MILF', 'BUT', 'SSN', 'FIFA', 'USD', 'CPU', 'AT', 'GG', 'Mar'}
new_words = {
'citron': -4.0,  
'hidenburg': -4.0,        
'moon': 4.0,
'highs': 2.0,
'mooning': 4.0,
'long': 2.0,
'short': -2.0,
'call': 4.0,
'calls': 4.0,    
'put': -4.0,
'puts': -4.0,    
'break': 2.0,
'tendie': 2.0,
'tendies': 2.0,
'town': 2.0,     
'overvalued': -3.0,
'undervalued': 3.0,
'buy': 4.0,
'sell': -4.0,
'gone': -1.0,
'gtfo': -1.7,
'paper': -1.7,
'bullish': 3.7,
'bearish': -3.7,
'bagholder': -1.7,
'stonk': 1.9,
'green': 1.9,
'money': 1.2,
'print': 2.2,
'rocket': 2.2,
'bull': 2.9,
'bear': -2.9,
'pumping': -1.0,
'sus': -3.0,
'offering': -2.3,
'rip': -4.0,
'downgrade': -3.0,
'upgrade': 3.0,     
'maintain': 1.0,          
'pump': 1.9,
'hot': 1.5,
'drop': -2.5,
'rebound': 1.5,  
'crack': 2.5,}




