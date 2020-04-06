sp_helptext '[dbo].[PopulateFrequencyMatrix]'

DROP PROCEDURE dbo.PopulateFrequencyMatrix
EXEC [dbo].[PopulateFrequencyMatrix]



SELECT * FROM [dbo].[FrequencyMatrix] WHERE All_Dates>300 AND Score IS NOT NULL ORDER BY Score*All_Dates DESC


SELECT * FROM [dbo].[FrequencyMatrix] WHERE All_Dates>300 ORDER BY All_Dates DESC


SELECT Sentiment, COUNT(*), SUM(All_Dates), SUM(Score)/COUNT(*)
FROM [dbo].[FrequencyMatrix]
GROUP BY Sentiment
ORDER BY SUM(All_Dates)



------------------------------------------------------------------------




SELECT AsOfDate, DJI, [S&P], VIX, EURUSD, MSFT, FB FROM 
(
	SELECT A.AsOfDate, A.Ticker, B.PriceAdjClose - A.PriceAdjClose AS DailyReturn
	FROM [dbo].[vwMarketData] A
	INNER JOIN (
		SELECT DATEADD(DAY, 1, AsOfDate) AS AsOfDate, PriceAdjClose, Ticker
		FROM [dbo].[vwMarketData]
			) B
	ON A.AsOfDate = B.AsOfDate
		AND A.Ticker = B.Ticker
	WHERE A.AsOfDate > (SELECT MIN([Date]) FROM [dbo].[MonthlyFreqData])
		AND A.AsOfDate < (SELECT MAX([Date]) FROM [dbo].[MonthlyFreqData])
) X
PIVOT
(
	Max(DailyReturn)
	FOR Ticker IN (DJI, [S&P], VIX, EURUSD, MSFT, FB)
)P








