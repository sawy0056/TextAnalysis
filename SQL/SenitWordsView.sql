

CREATE VIEW dbo.vwSentiWordsClean AS

SELECT Word1, MAX(PolarityScore) AS Score
FROM [TextAnalysis].[dbo].[SentiWords]
WHERE Word2='' 
GROUP BY Word1
