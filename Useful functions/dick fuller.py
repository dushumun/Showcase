""" Dick fuller stationarity test for python3.
	takes two positional arguments, the pandas dataframe containing the data and the speficic column used for testing stationarity
	"""

def test_stationarity(df, col):
    df['rolmean'] = df[col].rolling(window = 5, center=False).mean()
    df['rolstd'] = df[col].rolling(window = 5, center=False).std()
    
    plt.figure(figsize=(15,5))
    orig = plt.plot(df[col], color='blue',label='Original')
    mean = plt.plot(df['rolmean'], color='red', label='Rolling Mean')
    std = plt.plot(df['rolstd'], color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    print("Results of Dicker=Fuller Test:")
    dftest = adfuller(df[col], autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)