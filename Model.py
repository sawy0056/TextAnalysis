
### Libraries
import datetime
import pyodbc
import pandas as pd
import numpy as np
import scipy as sp
from scipy import linalg
from scipy.stats import norm
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot


###Plot Directory
plot_dir = 'C:\\Users\\sawy0\\Documents\\GitHub\\AMATH582\\FinalProject\\Figures\\'


### SQL Database Location
server = 'DESKTOP-BR9P1E0\SQLEXPRESS'
database = 'TextAnalysis'
dbTable = 'MonthlyFreqData'
cnxn = pyodbc.connect("DRIVER={SQL Server};SERVER="+server+";DATABASE="+database)


### If the web crawler has been run recently, update the Frequency Matrix in SQL
#sql = "EXEC [dbo].[PopulateFrequencyMatrix]"
#update_freq = pd.read_sql(sql,cnxn)


### Gather Frequency Data
sql = "SELECT * FROM [dbo].[FrequencyMatrix] WHERE abs(Score*All_Dates)>0 AND Score <> 0 ORDER BY Score*All_Dates"
raw_freq = pd.read_sql(sql,cnxn)
freq = raw_freq.copy()
del freq['Word']
del freq['Sentiment']
del freq['Score']
del freq['All_Dates']
freq = freq.fillna(0)
freq/freq.sum()
freq = freq.mul(raw_freq['Score'], axis=0)


### Gather Market Data
sql = """SELECT AsOfDate, DJI, [S&P], VIX, EURUSD, MSFT, FB FROM 
(
	SELECT C.AsOfDate, A.Ticker, B.PriceAdjClose - A.PriceAdjClose AS DailyReturn
	FROM dbo.Calendar C
	LEFT JOIN dbo.vwMarketData A
		ON C.AsOfDate = A.AsOfDate
	LEFT JOIN (
		SELECT DATEADD(DAY, 1, AsOfDate) AS AsOfDate, PriceAdjClose, Ticker
		FROM [dbo].[vwMarketData]
			) B
	ON A.AsOfDate = B.AsOfDate
		AND A.Ticker = B.Ticker
	WHERE C.AsOfDate >= (SELECT MIN([Date]) FROM [dbo].[MonthlyFreqData])
		AND C.AsOfDate <= (SELECT MAX([Date]) FROM [dbo].[MonthlyFreqData])
) X
PIVOT
(
	Max(DailyReturn)
	FOR Ticker IN (DJI, [S&P], VIX, EURUSD, MSFT, FB)
)P
ORDER BY AsOfDate"""


raw_mkt = pd.read_sql(sql,cnxn)
raw_mkt = raw_mkt.fillna(method='ffill') #Forward fill to cover weekend values


### Create a Buy Flag for classification purposes -1 = sell, 1 = buy
####################################################################
BuyFlag = raw_mkt.copy()
BuyFlag = BuyFlag.fillna(0)

for col in BuyFlag.columns[BuyFlag.columns!='AsOfDate']:
    BuyFlag[col][BuyFlag[col] < 0] = -1     #If negative, sell
    BuyFlag[col][BuyFlag[col] > 0] = 1      #If positive, buy
    BuyFlag[col][BuyFlag[col] == 0] = -1    #If 0, sell..... risk averse assumption


### Plot buy flag
fig = go.Figure(
        data = go.Heatmap(
        zmin = -1,
        zmax = 1,
        z=BuyFlag[BuyFlag.columns[BuyFlag.columns!='AsOfDate']].transpose(),
        x=list(BuyFlag['AsOfDate']),
        y=list(BuyFlag.columns[BuyFlag.columns!='AsOfDate']),
        colorscale='RdBu'
    )
)
plot(fig)


### 80 / 20 test train split
freq_tmp = freq.copy()
test_data = freq_tmp[freq_tmp.columns[78:98]]
train_data = freq_tmp[freq_tmp.columns[0:77]]


freq = train_data
##########################################################
### Analysis

df = pd.DataFrame(index=list(range(2,len(V))))
for ticker in raw_mkt.columns[raw_mkt.columns!='AsOfDate']:   
    accuracies = list()
    for num_features in range(2,len(V)):
        
        date_idx = BuyFlag['AsOfDate'].isin(freq.columns)
        mkt = BuyFlag[date_idx]
        mkt = mkt.reset_index()
        
        sort_idx = np.argsort(mkt[ticker])
        ordered_freq = freq[:][freq.columns[sort_idx]]
        ordered_signal = mkt.loc[sort_idx][ticker]
        
        ### SVD
        ##########################################################
        U, sigma, V = np.linalg.svd(ordered_freq, full_matrices = False)
        energy = np.power(sigma,1)/np.sum(np.power(sigma,1))
        var = np.power(sigma,2)/np.sum(np.power(sigma,2))
        np.cumsum(var)
        
        ###  Plot Cumulative Variance Explained
        ##########################################################
        import matplotlib.pyplot as plt
        %matplotlib qt
        plt.plot(np.cumsum(var))
        plt.title('Cumulative Variance Explained' )
        plt.ylabel('% Variance')
        plt.xlabel('Singular Value Number')
        #plt.show()
        plt.savefig(plot_dir + 'SVD_Variance_Explained' + ticker + '.png')
        
        ### Linear Discriminant Analysis
        ##########################################################
        SV = sigma*V.transpose()
        
        ### Locate buy and sell, truncate by num_features
        buy_idx = ordered_signal==1
        buy = SV[buy_idx,1:num_features]
        
        sell_idx = ordered_signal==-1
        sell = SV[sell_idx,1:num_features]
        
        buy_mean = np.mean(buy,axis=0)
        sell_mean = np.mean(sell,axis=0)
        
        ### Interclass Variance
        Sw = np.dot((buy-buy_mean).transpose(),(buy-buy_mean))
        Sw = Sw + np.dot((sell-sell_mean).transpose(),(sell-sell_mean))
        
        ### Between Class Variance
        Sb = np.outer((buy_mean-sell_mean).transpose(),(buy_mean-sell_mean))
        
        ### Eigen Decomposition
        e_val, e_vec = sp.linalg.eig(Sb, Sw)
        max_eval = np.max(np.abs(e_val))
        w = e_vec[:,e_val==max_eval]
        w = w/np.linalg.norm(w)
        
        #Project Buy and Sell onto new basis
        v_buy  = np.dot(w.transpose(), buy.transpose())
        v_sell = np.dot(w.transpose(), sell.transpose())
                
        #Draw Boundaries and calculate accuracy
        ##########################################################
        #Plot the new scatter
        %matplotlib qt
        val=0
        plt.plot(v_buy, np.zeros_like(v_buy) + val, 'o',color='green')
        plt.plot(v_sell, np.zeros_like(v_sell) + val, 'x',color='red')
        
        #Superimpose normal distribution fits
        buy_mu,  buy_std  = norm.fit(v_buy)
        sell_mu, sell_std = norm.fit(v_sell)
        x = np.linspace(xmin, xmax, 100)
        
        buy_p = norm.pdf(x, buy_mu, buy_std)
        sell_p = norm.pdf(x, sell_mu, sell_std)
        
        #Find Boundary
        pdf_sum = np.abs(buy_p-sell_p)
        if buy_mu<sell_mu:
            bound_idx = np.where(np.logical_and(x<sell_mu, x>buy_mu))
        else:
            bound_idx = np.where(np.logical_and(x>sell_mu, x<buy_mu))            
        pdf_sum_tmp = pdf_sum[bound_idx]
        x_tmp = x[bound_idx]
        x_tmp = x_tmp[pdf_sum_tmp==np.min(pdf_sum_tmp)]
        boundary = np.mean(x_tmp)
        
        plt.plot(x, buy_p, color='green', linewidth=2)
        plt.plot(x, sell_p, color='red', linewidth=2)
        
        ymin, ymax = plt.ylim()
        plt.vlines(boundary,ymin=ymin,ymax=ymax,color='black')
        
        if buy_mu<boundary:    
            buy_correct     = np.sum(v_buy<boundary)
            sell_correct    = np.sum(v_sell>boundary)
            total           = len(buy)+len(sell)    
            accuracy        = (buy_correct + sell_correct)/total
        else:
            buy_correct     = np.sum(v_buy>boundary)
            sell_correct    = np.sum(v_sell<boundary)
            total           = len(buy)+len(sell)    
            accuracy        = (buy_correct + sell_correct)/total
            
        accuracies.append(accuracy)
                
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.title(ticker + ' buy/sell is predicted with ' + str(np.round(accuracy*100,2)) + '% accuracy \n' + str(num_features) + ' features used in SVD')
        #plt.show()
        plt.savefig(plot_dir + 'LDA_Accuracy_' + ticker + '_' + str(num_features) +'.png')
        
    df[ticker] = pd.Series(accuracies, index=df.index)
  
#How does accuracy increase relative to num_features?
%matplotlib qt
plt.plot(df)
plt.title('LDA Accuracy by Number of Features from SVD')
plt.ylabel('% Accuracy')
plt.xlabel('Number of Principal Components Used')
plt.legend(df.columns)
plt.show()    
plt.savefig(plot_dir + 'LDA_Accuracy_ALL_By_Num_Feature.png')


##########################################################
### TEST DATA
##########################################################

df_test = pd.DataFrame(index=list(range(2,len(V))))
for ticker in raw_mkt.columns[raw_mkt.columns!='AsOfDate']:       
    test_accuracies = list()
    for num_features in range(2,len(V)):
    
        date_idx = BuyFlag['AsOfDate'].isin(test_data.columns)
        mkt = BuyFlag[date_idx]
        mkt = mkt.reset_index()
        
        sort_idx = np.argsort(mkt[ticker])
        ordered_freq = freq[:][freq.columns[sort_idx]]
        ordered_signal = mkt.loc[sort_idx][ticker]
                
        tmp = np.dot(U[:,0:(num_features-1)].transpose(), ordered_freq)
        pval = np.dot(w[0:(num_features-1)].transpose(),tmp)
                
        if buy_mu<boundary:    
            buy_correct     = np.sum(pval[:,ordered_signal==1]<boundary)
            sell_correct    = np.sum(pval[:,ordered_signal==-1]>boundary)
            total           = len(ordered_signal)    
            accuracy        = (buy_correct + sell_correct)/total
        else:
            buy_correct     = np.sum(pval[:,ordered_signal==1]>boundary)
            sell_correct    = np.sum(pval[:,ordered_signal==-1]<boundary)
            total           = len(ordered_signal)    
            accuracy        = (buy_correct + sell_correct)/total
        
        test_accuracies.append(accuracy)
    df_test[ticker] = pd.Series(test_accuracies, index=df.index)

#How does accuracy increase relative to num_features?
%matplotlib qt
plt.plot(df_test)
plt.title('LDA Accuracy by Number of Features from SVD')
plt.ylabel('% Accuracy')
plt.xlabel('Number of Principal Components Used')
plt.legend(df_test.columns)
plt.show()    
plt.savefig(plot_dir + 'Test_Accuracy_ALL_By_Num_Feature.png')











##########################################################
### Miscellaneous Plotting
##########################################################

### Heatmap of Raw Text and one indicator
##########################################################
fig = make_subplots(rows=2, cols=1, row_heights=[.1, .9]) #this a one cell subplot
fig.add_trace(
    go.Heatmap(
        zmin = -1,
        zmax = 1,
        z=ordered_freq,
        x=list(ordered_freq.columns),
        y=raw_freq['Word'],
        colorscale='RdBu'
    )
    ,row=2
    ,col=1
)
fig.add_trace(
    go.Scatter(
        mode='lines',
        #x=mkt.loc[sort_idx]['AsOfDate'], 
        y=ordered_signal,
        line=dict(color='black')
    )
    ,row=1
    ,col=1
)
plot(fig)


##########################################################
### Heatmap of Principal Orthogonal Components and one indicator
fig = make_subplots(rows=2, cols=1, row_heights=[.1, .9]) #this a one cell subplot
fig.add_trace(
    go.Heatmap(
        zmin = -1,
        zmax = 1,
        z = SV[buy_idx,1:5].transpose(),
        colorscale='RdBu'
    )
    ,row=2
    ,col=1
)
fig.add_trace(
    go.Scatter(
        mode='lines',
        y=ordered_signal,
        line=dict(color='black')
    )
    ,row=1
    ,col=1
)
plot(fig)

##########################################################
### Heatmap and one indicator
fig = make_subplots(rows=2, cols=1, row_heights=[.1, .9]) #this a one cell subplot
fig.add_trace(
    go.Heatmap(
        zmin = -1,
        zmax = 1,
        z=freq,
        x=list(freq.columns),
        y=raw_freq['Word'],
        colorscale='RdBu'
    )
#    ,secondary_y=False
    ,row=2
    ,col=1
)

fig.add_trace(
    go.Scatter(
        mode='lines',
        x=raw_mkt['AsOfDate'], 
        y=raw_mkt['DJI'],
        line=dict(color='black')
    )
#    ,secondary_y=True
    ,row=1
    ,col=1
)

plot(fig)


##########################################################
### Heatmap only
fig = go.Figure(data=go.Heatmap(
        zmin = -1,
        zmax = 1,
        z=freq,
        x=list(freq.columns),
        y=raw_freq['Word'],
        colorscale='RdBu'))        
        #https://plot.ly/python/builtin-colorscales/        

fig.update_layout(
    title='(Word Frequency) x (SentiWords Score)',
    xaxis_nticks=36)

plot(fig)

