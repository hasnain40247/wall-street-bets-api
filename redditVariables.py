import stockVariables
import praw

reddit = praw.Reddit(client_id='QaoL2TkHW67fgltj6AvrdA',
                     client_secret='t7tgeUoLNtr3CGmSLz6Er_AsMs8CLA',
                     user_agent='_lonelyPizza_')
subs = ['wallstreetbets']
# 'stocks', 'investing', 'stockmarket']     
post_flairs = {'Daily Discussion', 'Weekend Discussion', 'Discussion'}
goodAuth = {'AutoModerator'}   
uniqueCmt = True                
ignoreAuthP = {'example'}        
ignoreAuthC = {'example'}        

upvoteRatio = 0.70         
ups = 20       
limit = 10      
upvotes = 2     
picks = 10     
picks_ayz = 10  

posts, count, c_analyzed, tickers, titles, a_comments = 0, 0, 0, {}, [], {}
cmt_auth = {}


