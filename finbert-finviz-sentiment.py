import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from typing import List, Dict
from transformers import pipeline

# Configure finviz market new url and html class references

finviz_news_url = "https://finviz.com/news.ashx"
stock_news_table_class = 'styled-table-new'
news_link_column_class = 'news_link-cell'
news_date_cell_class = 'news_date-cell'


def download_finviz_market_news(max_headlines: int = 0,
                                include_metadata: bool = True) -> List[Dict]:
    """
    Download market news headlines from Finviz

    Args:
        max_headlines: Maximum number of headlines to fetch (0 to fetch all)
        include_metadata: Include timestamp and link metadata

    Returns:
        List of dictionaries containing headline metadata
    """

    # Headers similar to browsers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        response = requests.get(finviz_news_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        headlines = []

        # Find the news table
        news_table = soup.find('table', {'class': stock_news_table_class})

        # Find all tr under news table
        if news_table:
            rows = news_table.find_all('tr')

            # for each row
            for row in rows:
                if max_headlines > 0 and len(headlines) >= max_headlines:
                    break

                # Find link column
                link_cell = row.find('td', {'class': news_link_column_class})
                if not link_cell:
                    continue

                # Get headline text and link
                link_tag = link_cell.find('a')
                if link_tag:
                    headline_text = link_tag.get_text(strip=True)
                    headline_url = link_tag.get('href', '')

                    # Get timestamp if available
                    time_cell = row.find('td', {'class': news_date_cell_class})
                    timestamp = time_cell.get_text(strip=True) if time_cell else ''

                    headline_data = {
                        'headline': headline_text,
                        'fetched_at': datetime.now().isoformat()
                    }

                    if include_metadata:
                        headline_data.update({
                            'url': headline_url,
                            'timestamp': timestamp,
                        })

                    headlines.append(headline_data)

        return headlines

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except Exception as e:
        print(f"Error parsing data: {e}")
        return []


def get_headlines_as_text_list(headlines: List[Dict], max_headlines: int = 100) -> List[str]:
    """
    Function to get the headline texts as list
    """
    return [item['headline'] for item in headlines if item['headline']]


def save_headlines_to_csv(headlines: List[Dict], filename: str = 'finviz_headlines.csv',
                          max_headlines: int = 100) -> None:
    """
    Save to CSV file
    """
    if headlines:
        df = pd.DataFrame(headlines)
        df.to_csv(filename, index=False)
        print(f"Saved {len(headlines)} headlines to {filename}")
    else:
        print("Headlined is empty!")


# Sentiment analysis on finviz data
if __name__ == "__main__":

    # EGet headlines as text list
    print("Fetching news headlines with metadata...")
    headlines = download_finviz_market_news(0, True)
    print(f"\nTotal headlines: {len(headlines)}")
    print("First 5 headlines...")
    for item in headlines[:5]:
        print(f"Headline: {item['headline']}")
        print(f"Timestamp: {item.get('timestamp', 'N/A')}")
        print(f"URL: {item.get('url', 'N/A')}")
        print("-" * 40)

    # Example 3: Save to CSV
    save_headlines_to_csv(headlines, 'finviz_financial_news.csv', max_headlines=50)

    # Example 4: Integration with FinBERT sentiment analysis
    print("\n" + "=" * 80)
    print("Running FinBERT sentiment analysis on headlines...")

    try:
        # Initialize FinBERT
        classifier = pipeline("sentiment-analysis",
                              model="ProsusAI/finbert",
                              tokenizer="ProsusAI/finbert")

        # Classify
        headlines_text = get_headlines_as_text_list(headlines, max_headlines=20)

        for headline in headlines_text:
            if headline:  # Skip empty headlines
                sentiment = classifier(headline)[0]
                print(f"Headline: {headline[:60]}...")
                print(f"Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")
                print("-" * 60)

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
