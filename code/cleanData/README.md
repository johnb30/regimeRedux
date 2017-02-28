Data cleaning
=============

The three scripts in this directory focus on data prep/cleaning. The number
prepended to each filename indicates the order in which they should run.

Scraped data
------------

To clean the scraped data:

```
python 01-cleanCrew.py
```

which pulls in the JSON from the scraped data, performs text preprocessing,
and then adds a new field, `dataClean`, to the JSON. Data is saved to a
`data/cleaned` directory.

The second script is 

```
python 02-dataCrew.py
```

which takes in the cleaned data from the previous script, merged country-years 
together, and outputs files for analysis, including train-test splits, to the
`data/analysis` directory.
