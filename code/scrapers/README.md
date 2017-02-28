scrapers
========

This directory contains two web scrapers: one for Freedom House and one for
State Department reports.

Running
-------

To run either of the scrapers:

```
python SCRIPT_NAME.py OUTPUT_DIRECTORY
```

As a concrete example:

```
python state_scraper.py ../data/raw
```

would run the `state_scraper.py` and save the data to the directory `raw`
within the `data` directory.
