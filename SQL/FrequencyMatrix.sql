--- View to retrieve the pivoted word matrix
DECLARE @cols AS NVARCHAR(MAX),
		@query  AS NVARCHAR(MAX);

select	@cols = STUFF((SELECT ',' + QUOTENAME([Date]) 
			FROM dbo.MonthlyFreqData
			GROUP BY [Date]
			ORDER BY [Date]
            FOR XML PATH(''), TYPE
            ).value('.', 'NVARCHAR(MAX)') 
        ,1,1,'')

SET @query =	'SELECT Word, ' + @cols + ' FROM 
				(
					SELECT [Date], Word, Freq
					FROM dbo.MonthlyFreqData
				) A
				PIVOT 
				(
					SUM(Freq)
					FOR [Date] IN (' + @cols + ')
				) P 
				ORDER BY Word'

execute(@query)


