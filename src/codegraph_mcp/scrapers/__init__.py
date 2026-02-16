"""Scrapy-based web scrapers for external data sources.

Each site gets a spider inheriting from BaseSourceSpider.
Spiders yield dicts that get mapped to SourceDoc and stored in FalkorDB vectors.

Adding a new site:
  1. Create scrapers/newsite.py with a spider class inheriting BaseSourceSpider
  2. Override extract_item(response) to return site-specific fields
  3. Create indexers/newsite.py to wire spider → SourceIndex
  4. Add CLI command in cli.py
"""
