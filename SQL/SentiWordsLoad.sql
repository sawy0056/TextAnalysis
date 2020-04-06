DROP TABLE SentiWords
CREATE TABLE SentiWords (
	Lemma  varchar(max),
	PartOfSpeech  varchar(10),
	PolarityScore  numeric(12,9),
	Word1 varchar(max),
	Word2 varchar(max),
	Word3 varchar(max),
	Word4 varchar(max),
	Word5 varchar(max),
	Word6 varchar(max),
	Word7 varchar(max),
	Word8 varchar(max),
	Word9 varchar(max)
	)


BULK INSERT SentiWords FROM 'C:\Users\sawy0\Documents\GitHub\AMATH582\FinalProject\Sentiment3.csv' WITH(FIRSTROW = 2, 
FIELDTERMINATOR = ',', ROWTERMINATOR = '0x0a');




