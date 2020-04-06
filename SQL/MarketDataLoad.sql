--Load all market data that was downloaded from yahoo finance

---------------------------------------------------------------------------------------------
DROP TABLE mktDJI
CREATE TABLE mktDJI (
	AsOfDate Date,
	PriceOpen Numeric(30,8),
	PriceHigh Numeric(30,8),
	PriceLow Numeric(30,8),
	PriceClose Numeric(30,8),
	PriceAdjClose Numeric(30,8),
	Volume Numeric(30,8)
	)

BULK INSERT mktDJI FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Market Data\^DJI.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');


---------------------------------------------------------------------------------------------
DROP TABLE mktGSPC 
CREATE TABLE mktGSPC (
	AsOfDate Date,
	PriceOpen Numeric(30,8),
	PriceHigh Numeric(30,8),
	PriceLow Numeric(30,8),
	PriceClose Numeric(30,8),
	PriceAdjClose Numeric(30,8),
	Volume Numeric(30,8)
	)

BULK INSERT mktGSPC FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Market Data\^GSPC.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');


---------------------------------------------------------------------------------------------
DROP TABLE mktVIX
CREATE TABLE mktVIX (
	AsOfDate Date,
	PriceOpen Numeric(30,8),
	PriceHigh Numeric(30,8),
	PriceLow Numeric(30,8),
	PriceClose Numeric(30,8),
	PriceAdjClose Numeric(30,8),
	Volume Numeric(30,8)
	)

BULK INSERT mktVIX FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Market Data\^VIX.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');


---------------------------------------------------------------------------------------------
DROP TABLE mktEURUSD
CREATE TABLE mktEURUSD (
	AsOfDate Date,
	PriceOpen Numeric(30,8),
	PriceHigh Numeric(30,8),
	PriceLow Numeric(30,8),
	PriceClose Numeric(30,8),
	PriceAdjClose Numeric(30,8),
	Volume Numeric(30,8)
	)

BULK INSERT mktEURUSD FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Market Data\EURUSD=X.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');


---------------------------------------------------------------------------------------------
DROP TABLE mktFB
CREATE TABLE mktFB (
	AsOfDate Date,
	PriceOpen Numeric(30,8),
	PriceHigh Numeric(30,8),
	PriceLow Numeric(30,8),
	PriceClose Numeric(30,8),
	PriceAdjClose Numeric(30,8),
	Volume Numeric(30,8)
	)

BULK INSERT mktFB FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Market Data\FB.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');


---------------------------------------------------------------------------------------------
DROP TABLE mktMSFT
CREATE TABLE mktMSFT (
	AsOfDate Date,
	PriceOpen Numeric(30,8),
	PriceHigh Numeric(30,8),
	PriceLow Numeric(30,8),
	PriceClose Numeric(30,8),
	PriceAdjClose Numeric(30,8),
	Volume Numeric(30,8)
	)

BULK INSERT mktMSFT FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Market Data\MSFT.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');





