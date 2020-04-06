
CREATE VIEW dbo.vwMarketData AS

SELECT AsOfDate, 'DJI' AS 'Ticker', PriceOpen, PriceHigh, PriceLow, PriceClose, PriceAdjClose, Volume FROM [dbo].[mktDJI]
UNION
SELECT AsOfDate, 'S&P' AS 'Ticker', PriceOpen, PriceHigh, PriceLow, PriceClose, PriceAdjClose, Volume FROM [dbo].[mktGSPC]
UNION
SELECT AsOfDate, 'VIX' AS 'Ticker', PriceOpen, PriceHigh, PriceLow, PriceClose, PriceAdjClose, Volume FROM [dbo].[mktVIX]
UNION
SELECT AsOfDate, 'EURUSD' AS 'Ticker', PriceOpen, PriceHigh, PriceLow, PriceClose, PriceAdjClose, Volume FROM [dbo].[mktEURUSD]
UNION
SELECT AsOfDate, 'MSFT' AS 'Ticker', PriceOpen, PriceHigh, PriceLow, PriceClose, PriceAdjClose, Volume FROM [dbo].[mktMSFT]
UNION
SELECT AsOfDate, 'FB' AS 'Ticker', PriceOpen, PriceHigh, PriceLow, PriceClose, PriceAdjClose, Volume FROM [dbo].[mktFB]

