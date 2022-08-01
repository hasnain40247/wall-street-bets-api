from concurrent.futures import thread
from distutils.log import debug
import string
from urllib import request
from flask import Flask, url_for, request

#Helper Imports
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense 
import stockVariables
import pandas as pd
import redditVariables
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import squarify
from datetime import date
import requests
import plotly
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_datareader as pdr
import numpy as np


stock_prediction={}


app=Flask(__name__)

local_history=[]
def getVariables():
    for sub in redditVariables.subs:
        print(sub)
        subreddit = redditVariables.reddit.subreddit(sub)
        hot_python = subreddit.hot() 
        for submission in hot_python:
            flair = submission.link_flair_text
            author = submission.author.name
            if submission.upvote_ratio >= redditVariables.upvoteRatio and submission.ups > redditVariables.ups and (flair in redditVariables.post_flairs or flair is None) and author not in redditVariables.ignoreAuthP:
                submission.comment_sort = 'new'
                comments = submission.comments
            
                redditVariables.titles.append(submission.title)
                redditVariables.posts += 1
                submission.comments.replace_more(limit = redditVariables.limit)
                for comment in comments:
                    try:
                        auth = comment.author.name
                    except:
                        pass
                    redditVariables.c_analyzed += 1
                
                    if comment.score > redditVariables.upvotes and auth not in redditVariables.ignoreAuthC:
                        split = comment.body.split(' ')
                        for word in split:
                            word = word.replace("$", "")
                        
                       
                            if word.isupper() and len(word) <= 5 and word not in stockVariables.blacklist and word in stockVariables.stocks:
                                  
                                if redditVariables.uniqueCmt and auth not in redditVariables.goodAuth:
                                    try:
                                        if auth in redditVariables.cmt_auth[word]:
                                            break
                                    except:
                                        pass
                                
                                if word in redditVariables.tickers:
                                    redditVariables.tickers[word] += 1
                                    redditVariables.a_comments[word].append(comment.body)
                                    redditVariables.cmt_auth[word].append(auth)
                                    redditVariables.count += 1
                                else:
                                    redditVariables.tickers[word] = 1
                                    redditVariables.cmt_auth[word] = [auth]
                                    redditVariables.a_comments[word] = [comment.body]
                                    redditVariables.count += 1

    
def getFreq():
    tmp_stocks=[]
    tmp_frequency=[]
    for stock in redditVariables.tickers:
        tmp_stocks.append(stock)
        tmp_frequency.append(redditVariables.tickers[stock])
    plt.figure(figsize=(20,20))
  
    plt.xticks( rotation='vertical')    
    plt.bar(tmp_stocks,tmp_frequency)
    plt.savefig("./static/freq.png")


def getDataframe():
    symbols = dict(sorted(redditVariables.tickers.items(), key=lambda item: item[1], reverse = True))
    top_picks = list(symbols.keys())[0: redditVariables.picks]

    times = []
    top = []
    for i in top_picks:
        times.append(symbols[i])
        top.append(f"{i}: {symbols[i]}")

    scores, s = {}, {}

    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(stockVariables.new_words)

    picks_sentiment = list(symbols.keys())[0: redditVariables.picks_ayz]
    for symbol in picks_sentiment:
        stock_comments = redditVariables.a_comments[symbol]
        for cmnt in stock_comments:
            score = vader.polarity_scores(cmnt)
            if symbol in s:
                s[symbol][cmnt] = score
            else:
                s[symbol] = {cmnt: score}
            if symbol in scores:
                for key, _ in score.items():
                    scores[symbol][key] += score[key]
            else:
                scores[symbol] = score

        for key in score:
            scores[symbol][key] = scores[symbol][key] / symbols[symbol]
            scores[symbol][key] = "{pol:.3f}".format(pol=scores[symbol][key])

    df = pd.DataFrame(scores)
    df.index = ['Bearish', 'Neutral', 'Bullish', 'Total_Compound']
    df = df.T
    
    squarify.plot(sizes=times, label=top, alpha=0.7)
    plt.axis('off')
    plt.title(f"{redditVariables.picks} most mentioned picks")
    plt.savefig("./static/picks.png")
    df = df.astype(float)
    colors = ['red', 'springgreen', 'forestgreen', 'coral']
    df.plot(kind='bar', color=colors, title=f"Sentiment analysis of top {redditVariables.picks_ayz} picks:")
    plt.savefig("./static/sentiment.png")

    df['date'] = [date.today() for x in range(df.shape[0])]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'stock'}, inplace=True)

    df.to_csv('./static/df.csv')

    return symbols,top_picks

def getCurrent():
    df = pd.read_csv('./static/df.csv', index_col=0)

    client_id = "OKH4HH1ZQCNKWBTNARXZGPKWXFFVRAF3" 
    stocks_list = list(df['stock'])

    parameters_quotes = {
        'apikey': client_id,
        'symbol': stocks_list,
    }

    parameters_fund = {
        'apikey': client_id,
        'symbol': stocks_list,
        'projection': 'fundamental'
    }

    quotes_url = f'https://api.tdameritrade.com/v1/marketdata/quotes?apikey={client_id}'
    fundamental_url = f'https://api.tdameritrade.com/v1/instruments?apikey={client_id}'


    data_quotes = requests.get(url = quotes_url, params = parameters_quotes).json()
    data_fundamental = requests.get(url = fundamental_url, params = parameters_fund).json()
    df_q = pd.DataFrame.from_dict(data_quotes, orient = 'index')
    df_q.reset_index(inplace=True)
    df_q.drop('index', axis=1, inplace=True)

    fund_cols = []
    for column in [x for x in [*data_fundamental[stocks_list[0]]['fundamental']]]:
        fund_cols.append(column)


    df_f = pd.DataFrame(columns = fund_cols)
    for stock in stocks_list:
        df_f = df_f.append(pd.Series(data_fundamental[stock]['fundamental'], index=fund_cols), ignore_index=True)


    current = pd.concat([df, df_q, df_f], axis=1)
    current = current.loc[:, ~current.columns.duplicated()]

    return current



def getHistoricData(current):
    historic_sentiment_analysis = pd.read_csv('./static/historic_sentiment_analysis.csv')
    historic_sentiment_analysis = pd.concat([historic_sentiment_analysis, current], axis = 0, ignore_index=True)
    historic_sentiment_analysis.drop_duplicates()
    historic_sentiment_analysis.to_csv('./static/historic_sentiment_analysis.csv', index=False)
    
    historic_sentiment_analysis = pd.read_csv('./static/historic_sentiment_analysis.csv')
    historic_sentiment_analysis['date'] = pd.to_datetime(historic_sentiment_analysis['date'])


    plot_json={}

   
    sentiment_list = ['Bullish', 'Bearish', 'Neutral', 'Total_Compound']

    for sentiment in sentiment_list:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for stock in historic_sentiment_analysis['stock'].unique():
        

            fig.add_trace(go.Line(x=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock]['date'],
                                y=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock][sentiment],
                                mode='lines+markers',
                                name=f'{stock}'),secondary_y=False)


        fig.update_layout(title=f'WSB {sentiment} Sentiment',
                        xaxis_title='Date',
                        yaxis_title='Sentiment Score')
        fig.update_xaxes(title_text="Date")
 

        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)

       
        plot_json[sentiment]=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json



def getIndivStock():
    plot_json={}
    

    historic_sentiment_analysis = pd.read_csv('./static/historic_sentiment_analysis.csv')
    
    sentiment_list = ['Bullish', 'Bearish', 'Neutral', 'Total_Compound']

    for stock in historic_sentiment_analysis['stock'].unique():
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Line(x=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock]['date'],
                                y=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock][sentiment_list[0]],
                                mode='lines+markers', fill='tozeroy',     line_color='green', 

                                name='Positive'),secondary_y=False)
        fig.add_trace(go.Line(x=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock]['date'],
                                y=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock][sentiment_list[1]],
                                mode='lines+markers', fill="tozeroy",     line_color='red',

                                name='Negative'),secondary_y=False)
        fig.add_trace(go.Line(x=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock]['date'],
                                y=historic_sentiment_analysis[historic_sentiment_analysis['stock'] == stock]['closePrice'],
                                mode='lines+markers',     line_color='yellow',
                                name='Prices'),secondary_y=True)

        fig.update_layout(title=f'WSB {stock} Sentiment',
                        xaxis_title='Date',
                        yaxis_title='Sentiment Score')
        fig.update_xaxes(title_text="Date")
 

        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)


        plot_json[stock]=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json
        

def getHistory():
    current=pd.read_csv('./static/df.csv', index_col=0)
    plot_json={}
    data_source = 'yahoo'
    start_date = '2021-07-20'
    end_date = '2022-07-22'
    
    personal_list=[]
    stock_list=[]
    for item in np.array(current["stock"]):
        stock_prediction[item]= pdr.DataReader(item, data_source, start_date, end_date)
        personal_list.append(item)
    for item in stock_prediction:
        stock_list.append(stock_prediction[item])
    
    hist_analytics = pd.concat(stock_list, axis=1, keys= np.array(current["stock"]) )
    hist_analytics.columns.names = ['Stock Ticker', 'Stock Info']
    hist_analytics.to_csv("./static/hist_analytics.csv")
    c = hist_analytics.xs(key='Close', axis=1, level='Stock Info')
    c
    for item in np.array(current["stock"]):
        
        fig = go.Figure(data=[go.Candlestick(x=hist_analytics.index, 
                open=hist_analytics[item]['Open'],
                high = hist_analytics[item]['High'],
                low = hist_analytics[item]['Low'],
                close = hist_analytics[item]['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        plot_json[item]=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return plot_json,personal_list




def getPrediction():
    
    personal_list=[]

    for item in stock_prediction:
        personal_list.append(item)

    return personal_list

def getPredict(symbol):
    current=pd.read_csv("./static/df.csv")
    print(np.array(current["stock"]))

    
    stock_list=[]
    personal_list=[]
    plot_json={}

    for item in stock_prediction:
        stock_list.append(stock_prediction[item])
        personal_list.append(item)

    hist_analytics = pd.concat(stock_list, axis=1, keys= np.array(current["stock"]) )
    hist_analytics.columns.names = ['Stock Ticker', 'Stock Info']


    df_stock = hist_analytics.xs(key=symbol, axis=1, level='Stock Ticker')
    train, test = df_stock.iloc[0:-50], df_stock.iloc[-50:len(df_stock)]
    print(len(train), len(test))

    train_max = train.max()
    train_min = train.min()

    train = (train - train_min)/(train_max - train_min)
    test = (test - train_min)/(train_max - train_min)

    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)


    time_steps = 10

    X_train, y_train = create_dataset(train, train.Close, time_steps)
    X_test, y_test = create_dataset(test, test.Close, time_steps)


    model = Sequential()
    model.add(LSTM(250, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        shuffle=False
    )

    y_pred = model.predict(X_test)
    y_test = y_test*(train_max[0] - train_min[0]) + train_min[0]
    y_pred = y_pred*(train_max[0] - train_min[0]) + train_min[0]
    y_train = y_train*(train_max[0] - train_min[0]) + train_min[0]

    print(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_train), len(y_train) + len(y_test)), y=y_test.flatten(),
                    mode='lines',
                    name='True'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_train), len(y_train) + len(y_test)), y=y_pred.flatten(),
                    mode='lines',
                    name='Prediction'))
    fig.add_trace(go.Scatter(x=np.arange(0, len(y_train)),y= y_train.flatten(),
                    mode='lines',
                    name='History'))
    plot_json=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plot_json,round(np.mean(y_pred.flatten()),2)


    
    
@app.route("/prediction")
def prediction():
    stocks=getPrediction()

    return {"stocks": stocks}



@app.route("/predict")
def predict():
    symbol=request.args.get('symbol')
    print(symbol)
    plot_json,price=getPredict(symbol)
    print(plot_json)
    print(str(price))

    return {"plot_json": plot_json,"price": str(price)}






@app.route("/indivstock")
def indivstock():

    plot_json=getIndivStock()
   

    

    return {"plot_json": plot_json }


@app.route("/histanal")
def histanal():

    plot_json,stock_list=getHistory()
   

    

    return {"plot_json": plot_json,"stock_list": stock_list }




@app.route("/members")
def members():
  

    getVariables()
   

    getFreq()
    

    symbols,top_picks=getDataframe()

    df=getCurrent()
    plot_json=getHistoricData(df)
    tables=[df.to_html(classes='data', header="true")]

    return { 
        "plot_json":plot_json,
        "statistics":{
        "symbols": symbols,
        "top_picks": top_picks,
        "posts": redditVariables.posts,
        "no_comments": redditVariables.c_analyzed,
        "tickers": redditVariables.tickers,
        "all_comments": redditVariables.a_comments
    }, "table": tables  }

if __name__ == "__main__":
    app.run(threaded=True, port=5000)