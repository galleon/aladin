"""
Web crawling: crawl4ai, URL filtering, table extraction.
"""
from __future__ import annotations

import logging
import re
from typing import List, Set
from urllib.parse import urljoin

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from shared.config import settings

logger = logging.getLogger(__name__)


def url_matches_pattern(url: str, pattern: str) -> bool:
    """Check if URL matches a glob-like pattern."""
    regex_pattern = pattern.replace("*", ".*")
    return bool(re.search(regex_pattern, url))


def matches_patterns(
    url: str,
    inclusion_patterns: List[str],
    exclusion_patterns: List[str],
) -> bool:
    """Check if URL matches inclusion/exclusion patterns."""
    for pattern in exclusion_patterns:
        if url_matches_pattern(url, pattern):
            return False
    if not inclusion_patterns:
        return True
    for pattern in inclusion_patterns:
        if url_matches_pattern(url, pattern):
            return True
    return False


def extract_tables(html: str, strategy: str) -> List[dict]:
    """Extract tables from HTML content."""
    tables = []
    if not html:
        return tables
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for i, table in enumerate(soup.find_all("table")):
            table_data = {"index": i, "html": str(table)}
            if strategy == "markdown":
                table_data["markdown"] = table_to_markdown(table)
            elif strategy == "summary_and_html":
                table_data["markdown"] = table_to_markdown(table)
                table_data["needs_summary"] = True
            tables.append(table_data)
    except Exception as e:
        logger.warning("Failed to extract tables: %s", e)
    return tables


def table_to_markdown(table) -> str:
    """Convert HTML table to markdown."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    if len(rows) > 1:
        header_len = rows[0].count("|") - 1
        separator = "|" + "|".join(["---"] * header_len) + "|"
        rows.insert(1, separator)
    return "\n".join(rows)


async def crawl_pages(
    start_url: str,
    source_config: dict,
    processing_config: dict,
    metrics_callback=None,
) -> List[dict]:
    """
    Crawl web pages using crawl4ai.

    Returns:
        List of page data dicts with url, depth, title, markdown, html, links, tables.
    """
    pages = []
    visited: Set[str] = set()
    queue = [(start_url, 0)]

    depth_limit = source_config.get("depth_limit", settings.DEFAULT_CRAWL_DEPTH)
    max_pages = source_config.get("max_pages", 100)
    strategy = source_config.get("strategy", "bfs")
    inclusion_patterns = source_config.get("inclusion_patterns", [])
    exclusion_patterns = source_config.get("exclusion_patterns", [])

    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for=processing_config.get("wait_for_selector"),
        page_timeout=processing_config.get("wait_timeout", 30) * 1000,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        while queue and len(pages) < max_pages:
            url, depth = queue.pop(0) if strategy == "bfs" else queue.pop()
            url = url.rstrip("/")

            if url in visited or depth > depth_limit:
                continue
            if not matches_patterns(url, inclusion_patterns, exclusion_patterns):
                continue

            visited.add(url)
            try:
                logger.info("Crawling: %s (depth %d)", url, depth)
                result = await crawler.arun(url=url, config=crawler_config)

                if result.success:
                    page_data = {
                        "url": url,
                        "depth": depth,
                        "title": result.metadata.get("title", ""),
                        "markdown": result.markdown,
                        "html": result.html if processing_config.get("extract_tables") else None,
                        "links": result.links.get("internal", []) if hasattr(result, "links") else [],
                    }
                    if processing_config.get("extract_tables"):
                        page_data["tables"] = extract_tables(
                            result.html,
                            processing_config.get("table_strategy", "markdown"),
                        )
                    pages.append(page_data)
                    if metrics_callback:
                        metrics_callback(1)

                    if depth < depth_limit:
                        for link in page_data.get("links", []):
                            link_url = link.get("href", "") if isinstance(link, dict) else str(link)
                            if link_url:
                                absolute_url = urljoin(url, link_url)
                                if absolute_url not in visited:
                                    queue.append((absolute_url, depth + 1))
                else:
                    logger.warning("Failed to crawl %s: %s", url, result.error_message)
            except Exception as e:
                logger.error("Error crawling %s: %s", url, e)

    return pages
                        link_url = link.get("href", "") if isinstance(link, dict) else str(link)
                        if link_url:
                            absolute_url = urljoin(url, link_url)
                            if absolute_url not in visited:
                                queue.append((absolute_url, depth + 1))
            except Exception as e:
                logger.error("Error crawling %s: %s", url, e)

    return pages
