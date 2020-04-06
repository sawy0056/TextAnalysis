CREATE PROCEDURE PopulateFrequencyMatrix AS


--If exists, drop FrequencyMatrix Table
IF OBJECT_ID('dbo.FrequencyMatrix', 'U') IS NOT NULL 
DROP TABLE dbo.FrequencyMatrix



--- View to retrieve the pivoted word matrix
DECLARE @cols AS NVARCHAR(MAX),
		@query  AS NVARCHAR(MAX);

select	@cols = 'All_Dates, ' + STUFF((SELECT ',' + QUOTENAME([Date]) 
								FROM dbo.MonthlyFreqData
								GROUP BY [Date]
								ORDER BY [Date]
								FOR XML PATH(''), TYPE
								).value('.', 'NVARCHAR(MAX)') 
							,1,1,'')

SET @query =	'SELECT Word, Sentiment, Score, ' + @cols +  ' 
				INTO dbo.FrequencyMatrix 
				FROM 
				(
					SELECT [Date], Word, Freq, Score,
						CASE WHEN Score>0 THEN ''Positive''
								WHEN Score<0 THEN ''Negative''
								WHEN Score=0 THEN ''Neutral''
								ELSE ''Unknown''
						END AS ''Sentiment''
					FROM (
						SELECT [Date], Word, Freq
						FROM dbo.MonthlyFreqData A
						UNION ALL
						SELECT ''All_Dates'' AS [Date], Word, SUM(Freq) AS Freq
						FROM dbo.MonthlyFreqData A
						GROUP BY Word
						) A
					LEFT JOIN dbo.vwSentiWordsClean B
					ON TRIM(LOWER(A.Word)) = TRIM(LOWER(B.Word1))
				) A
				PIVOT 
				(
					SUM(Freq)
					FOR [Date] IN (' + @cols + ')
				) P 
				ORDER BY Score'

execute(@query)
GO

