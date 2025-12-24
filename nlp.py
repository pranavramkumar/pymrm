"""
Natural Language Processing for ESG Sentiment Analysis

This module provides comprehensive NLP functionality for ESG (Environmental, Social, Governance) sentiment analysis including:
- News and social media data scraping and collection
- Text preprocessing: tokenization, stemming, stopword removal
- Document term matrices with TF-IDF vectorization
- Visualization: word clouds, feature co-occurrence networks
- Topic modeling using LDA and advanced techniques
- Multi-dimensional sentiment analysis for ESG themes
- Sentiment signal generation and scoring
- Named Entity Recognition for ESG-relevant entities
- Time-series sentiment analysis and trend detection

Author: Claude AI
Created: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
from collections import Counter, defaultdict
import logging

# NLP and Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy

# Scikit-learn for ML and text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sentiment Analysis
from textblob import TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("VADER sentiment not available. Install with: pip install vaderSentiment")
    SentimentIntensityAnalyzer = None

# Visualization
from wordcloud import WordCloud
import networkx as nx

# Web scraping
from bs4 import BeautifulSoup
import feedparser

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp_spacy = None


@dataclass
class NLPConfig:
    """Configuration class for NLP processing."""

    # General settings
    precision: int = 6
    max_features: int = 1000
    min_df: int = 2
    max_df: float = 0.95

    # Text preprocessing
    remove_stopwords: bool = True
    use_stemming: bool = True
    use_lemmatization: bool = False
    min_word_length: int = 3
    max_word_length: int = 20

    # ESG keywords
    esg_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        'environmental': [
            'climate', 'carbon', 'emissions', 'sustainability', 'renewable', 'energy',
            'pollution', 'waste', 'recycling', 'green', 'environmental', 'biodiversity',
            'deforestation', 'water', 'air quality', 'ecosystem', 'greenhouse gas'
        ],
        'social': [
            'diversity', 'inclusion', 'equality', 'human rights', 'labor', 'employee',
            'community', 'safety', 'health', 'welfare', 'discrimination', 'workplace',
            'social responsibility', 'stakeholder', 'philanthropy', 'education'
        ],
        'governance': [
            'governance', 'transparency', 'accountability', 'ethics', 'compliance',
            'board', 'executive', 'audit', 'risk management', 'disclosure',
            'shareholder', 'voting', 'compensation', 'corruption', 'bribery'
        ]
    })

    # Sentiment analysis
    sentiment_threshold: float = 0.1
    confidence_threshold: float = 0.6

    # Topic modeling
    n_topics: int = 10
    topic_words: int = 10

    # Data collection
    max_articles: int = 100
    date_range_days: int = 30

    # API settings
    news_api_delay: float = 1.0
    twitter_api_delay: float = 0.5

    # Visualization
    wordcloud_width: int = 800
    wordcloud_height: int = 400
    figure_size: Tuple[int, int] = (15, 10)
    color_palette: str = 'viridis'


class DataScraper:
    """
    Web scraping functionality for news and social media data.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_news_articles(self, companies: List[str], sources: List[str] = None) -> Dict[str, Any]:
        """
        Scrape news articles related to ESG topics for given companies.

        Parameters:
        -----------
        companies : List[str]
            List of company names or tickers
        sources : List[str]
            List of news source URLs or RSS feeds

        Returns:
        --------
        Dict with scraped articles and metadata
        """
        try:
            if sources is None:
                sources = self._get_default_news_sources()

            articles = []

            for company in companies:
                for source in sources:
                    try:
                        company_articles = self._scrape_single_source(company, source)
                        articles.extend(company_articles)
                        time.sleep(self.config.news_api_delay)
                    except Exception as e:
                        logger.warning(f"Failed to scrape {source} for {company}: {e}")
                        continue

            # Remove duplicates
            unique_articles = self._remove_duplicate_articles(articles)

            # Filter for ESG relevance
            esg_articles = self._filter_esg_articles(unique_articles)

            return {
                'articles': esg_articles,
                'total_scraped': len(articles),
                'unique_articles': len(unique_articles),
                'esg_relevant': len(esg_articles),
                'companies': companies,
                'scraping_date': datetime.now(),
                'sources_used': sources
            }

        except Exception as e:
            return {'error': str(e)}

    def scrape_social_media(self, companies: List[str], platforms: List[str] = None) -> Dict[str, Any]:
        """
        Scrape social media posts related to ESG topics.

        Parameters:
        -----------
        companies : List[str]
            List of company names or handles
        platforms : List[str]
            Social media platforms to scrape

        Returns:
        --------
        Dict with scraped social media posts
        """
        try:
            if platforms is None:
                platforms = ['twitter', 'reddit']

            posts = []

            for company in companies:
                for platform in platforms:
                    try:
                        if platform.lower() == 'twitter':
                            platform_posts = self._scrape_twitter_alternative(company)
                        elif platform.lower() == 'reddit':
                            platform_posts = self._scrape_reddit(company)
                        else:
                            logger.warning(f"Platform {platform} not supported")
                            continue

                        posts.extend(platform_posts)
                        time.sleep(self.config.twitter_api_delay)

                    except Exception as e:
                        logger.warning(f"Failed to scrape {platform} for {company}: {e}")
                        continue

            # Filter for ESG relevance
            esg_posts = self._filter_esg_posts(posts)

            return {
                'posts': esg_posts,
                'total_scraped': len(posts),
                'esg_relevant': len(esg_posts),
                'companies': companies,
                'platforms': platforms,
                'scraping_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def get_financial_news_feeds(self, esg_focused: bool = True) -> Dict[str, Any]:
        """
        Get financial news from RSS feeds with ESG focus.

        Parameters:
        -----------
        esg_focused : bool
            Whether to focus on ESG-related feeds

        Returns:
        --------
        Dict with news feed data
        """
        try:
            if esg_focused:
                feeds = [
                    'https://www.reuters.com/business/sustainable-business/rss',
                    'https://www.bloomberg.com/feeds/sustainability',
                    'https://feeds.feedburner.com/greenbiz',
                    'https://www.sustainable-finance.org/feed/',
                ]
            else:
                feeds = [
                    'https://feeds.reuters.com/reuters/businessNews',
                    'https://feeds.bloomberg.com/bloomberg/news',
                    'https://www.ft.com/rss/companies',
                ]

            all_articles = []

            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)

                    for entry in feed.entries[:self.config.max_articles // len(feeds)]:
                        article = {
                            'title': entry.get('title', ''),
                            'description': entry.get('description', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': feed.feed.get('title', 'Unknown'),
                            'feed_url': feed_url
                        }
                        all_articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to parse feed {feed_url}: {e}")
                    continue

            # Filter for ESG content
            esg_articles = self._filter_esg_articles(all_articles)

            return {
                'articles': esg_articles,
                'total_articles': len(all_articles),
                'esg_articles': len(esg_articles),
                'feeds_processed': len(feeds),
                'collection_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_default_news_sources(self) -> List[str]:
        """Get default news sources for scraping."""
        return [
            'https://www.reuters.com',
            'https://www.bloomberg.com',
            'https://www.ft.com',
            'https://www.wsj.com',
            'https://www.marketwatch.com'
        ]

    def _scrape_single_source(self, company: str, source: str) -> List[Dict[str, Any]]:
        """Scrape a single news source for company-related articles."""
        try:
            # Construct search URL (simplified approach)
            search_terms = f"{company} ESG sustainability"

            # For demonstration, we'll use a simplified scraping approach
            # In practice, you'd need to adapt to each site's structure

            articles = []

            # Simulate article scraping (replace with actual implementation)
            for i in range(5):  # Simulate finding 5 articles
                article = {
                    'title': f"Sample ESG article {i+1} about {company}",
                    'content': f"Sample content discussing {company}'s ESG initiatives and sustainability efforts.",
                    'url': f"{source}/article-{i+1}",
                    'date': datetime.now() - timedelta(days=i),
                    'source': source,
                    'company': company
                }
                articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"Error scraping {source}: {e}")
            return []

    def _scrape_twitter_alternative(self, company: str) -> List[Dict[str, Any]]:
        """Scrape Twitter-like data using alternative methods."""
        try:
            # Since Twitter API requires authentication, we'll simulate data
            # In practice, you'd use official API or alternative services

            posts = []

            for i in range(10):  # Simulate 10 posts
                post = {
                    'text': f"Sample tweet about {company} ESG practices and sustainability initiatives.",
                    'author': f"user_{i}",
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'platform': 'twitter',
                    'company': company,
                    'engagement': {'likes': np.random.randint(0, 100), 'retweets': np.random.randint(0, 50)}
                }
                posts.append(post)

            return posts

        except Exception as e:
            logger.error(f"Error scraping Twitter for {company}: {e}")
            return []

    def _scrape_reddit(self, company: str) -> List[Dict[str, Any]]:
        """Scrape Reddit posts related to company ESG."""
        try:
            # Simulate Reddit data
            posts = []

            for i in range(5):  # Simulate 5 Reddit posts
                post = {
                    'title': f"Discussion about {company}'s ESG score",
                    'text': f"Detailed discussion about {company}'s environmental and social governance practices.",
                    'subreddit': 'investing',
                    'author': f"reddit_user_{i}",
                    'timestamp': datetime.now() - timedelta(days=i),
                    'platform': 'reddit',
                    'company': company,
                    'score': np.random.randint(-10, 100)
                }
                posts.append(post)

            return posts

        except Exception as e:
            logger.error(f"Error scraping Reddit for {company}: {e}")
            return []

    def _remove_duplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        try:
            if not articles:
                return articles

            unique_articles = []
            seen_titles = set()

            for article in articles:
                title = article.get('title', '').lower().strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)

            return unique_articles

        except Exception:
            return articles

    def _filter_esg_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles for ESG relevance."""
        try:
            esg_articles = []

            # Combine all ESG keywords
            all_esg_keywords = []
            for category in self.config.esg_keywords.values():
                all_esg_keywords.extend(category)

            for article in articles:
                # Check title and content for ESG keywords
                text = f"{article.get('title', '')} {article.get('content', '')} {article.get('description', '')}".lower()

                # Count ESG keyword matches
                esg_score = sum(1 for keyword in all_esg_keywords if keyword.lower() in text)

                if esg_score > 0:
                    article['esg_score'] = esg_score
                    article['esg_categories'] = self._identify_esg_categories(text)
                    esg_articles.append(article)

            return esg_articles

        except Exception:
            return articles

    def _filter_esg_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter social media posts for ESG relevance."""
        return self._filter_esg_articles(posts)  # Same logic

    def _identify_esg_categories(self, text: str) -> List[str]:
        """Identify which ESG categories are mentioned in text."""
        categories = []

        for category, keywords in self.config.esg_keywords.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                categories.append(category)

        return categories


class TextPreprocessor:
    """
    Text preprocessing and feature extraction.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Add custom ESG stopwords
        self.stop_words.update(['company', 'business', 'corporate', 'firm', 'organization'])

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with tokenization, cleaning, stemming/lemmatization.

        Parameters:
        -----------
        text : str
            Raw text to preprocess

        Returns:
        --------
        str
            Preprocessed text
        """
        try:
            if not text or not isinstance(text, str):
                return ""

            # Convert to lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize
            tokens = word_tokenize(text)

            # Filter tokens
            processed_tokens = []
            for token in tokens:
                # Skip if too short or too long
                if len(token) < self.config.min_word_length or len(token) > self.config.max_word_length:
                    continue

                # Skip stopwords if enabled
                if self.config.remove_stopwords and token in self.stop_words:
                    continue

                # Apply stemming or lemmatization
                if self.config.use_stemming:
                    token = self.stemmer.stem(token)
                elif self.config.use_lemmatization:
                    token = self.lemmatizer.lemmatize(token)

                processed_tokens.append(token)

            return ' '.join(processed_tokens)

        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return ""

    def create_document_term_matrix(self, documents: List[str], method: str = 'tfidf') -> Dict[str, Any]:
        """
        Create document-term matrix using TF-IDF or count vectorization.

        Parameters:
        -----------
        documents : List[str]
            List of preprocessed documents
        method : str
            Vectorization method ('tfidf' or 'count')

        Returns:
        --------
        Dict with document-term matrix and metadata
        """
        try:
            if not documents:
                return {'error': 'No documents provided'}

            # Preprocess documents
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            processed_docs = [doc for doc in processed_docs if doc.strip()]  # Remove empty docs

            if not processed_docs:
                return {'error': 'No valid documents after preprocessing'}

            if method == 'tfidf':
                vectorizer = TfidfVectorizer(
                    max_features=self.config.max_features,
                    min_df=self.config.min_df,
                    max_df=self.config.max_df,
                    ngram_range=(1, 2)  # Include bigrams
                )
            elif method == 'count':
                vectorizer = CountVectorizer(
                    max_features=self.config.max_features,
                    min_df=self.config.min_df,
                    max_df=self.config.max_df,
                    ngram_range=(1, 2)
                )
            else:
                return {'error': f'Unsupported method: {method}'}

            # Fit and transform documents
            dtm = vectorizer.fit_transform(processed_docs)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Convert to DataFrame for easier analysis
            dtm_df = pd.DataFrame(dtm.toarray(), columns=feature_names)

            # Calculate document statistics
            doc_stats = self._calculate_document_statistics(dtm_df, processed_docs)

            # Feature importance
            feature_importance = self._calculate_feature_importance(dtm_df, method)

            return {
                'document_term_matrix': dtm_df,
                'vectorizer': vectorizer,
                'feature_names': feature_names.tolist(),
                'method': method,
                'document_statistics': doc_stats,
                'feature_importance': feature_importance,
                'matrix_shape': dtm.shape,
                'processed_documents': processed_docs,
                'creation_date': datetime.now()
            }

        except Exception as e:
            return {'error': str(e)}

    def extract_ngrams(self, text: str, n: int = 2, top_k: int = 20) -> Dict[str, Any]:
        """
        Extract top n-grams from text.

        Parameters:
        -----------
        text : str
            Text to analyze
        n : int
            N-gram size
        top_k : int
            Number of top n-grams to return

        Returns:
        --------
        Dict with n-gram analysis
        """
        try:
            processed_text = self.preprocess_text(text)
            tokens = processed_text.split()

            if len(tokens) < n:
                return {'error': f'Text too short for {n}-grams'}

            # Generate n-grams
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)

            # Count frequencies
            ngram_counts = Counter(ngrams)
            top_ngrams = ngram_counts.most_common(top_k)

            return {
                'ngrams': top_ngrams,
                'total_ngrams': len(ngrams),
                'unique_ngrams': len(ngram_counts),
                'n': n,
                'coverage': sum(count for _, count in top_ngrams) / len(ngrams) if ngrams else 0
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_document_statistics(self, dtm_df: pd.DataFrame, documents: List[str]) -> Dict[str, Any]:
        """Calculate statistics about the document collection."""
        try:
            stats = {
                'num_documents': len(documents),
                'num_features': dtm_df.shape[1],
                'avg_doc_length': np.mean([len(doc.split()) for doc in documents]),
                'vocab_size': len(dtm_df.columns),
                'sparsity': (dtm_df == 0).sum().sum() / (dtm_df.shape[0] * dtm_df.shape[1])
            }
            return stats
        except Exception:
            return {}

    def _calculate_feature_importance(self, dtm_df: pd.DataFrame, method: str) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            if method == 'tfidf':
                # Sum TF-IDF scores across documents
                importance = dtm_df.sum(axis=0).sort_values(ascending=False)
            else:
                # Sum counts across documents
                importance = dtm_df.sum(axis=0).sort_values(ascending=False)

            # Return top features
            top_features = importance.head(20)
            return {feature: round(score, self.config.precision) for feature, score in top_features.items()}

        except Exception:
            return {}


class VisualizationTools:
    """
    Text visualization tools including word clouds and co-occurrence networks.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()

    def create_wordcloud(self, text: str, title: str = "Word Cloud",
                        esg_category: str = None) -> Dict[str, Any]:
        """
        Create word cloud visualization.

        Parameters:
        -----------
        text : str
            Text for word cloud
        title : str
            Title for the visualization
        esg_category : str
            ESG category for color coding

        Returns:
        --------
        Dict with word cloud information
        """
        try:
            if not text or not text.strip():
                return {'error': 'No text provided for word cloud'}

            # Preprocess text
            preprocessor = TextPreprocessor(self.config)
            processed_text = preprocessor.preprocess_text(text)

            if not processed_text.strip():
                return {'error': 'No valid words after preprocessing'}

            # Create color map based on ESG category
            if esg_category:
                color_map = {
                    'environmental': 'Greens',
                    'social': 'Blues',
                    'governance': 'Oranges'
                }
                colormap = color_map.get(esg_category.lower(), 'viridis')
            else:
                colormap = 'viridis'

            # Generate word cloud
            wordcloud = WordCloud(
                width=self.config.wordcloud_width,
                height=self.config.wordcloud_height,
                background_color='white',
                colormap=colormap,
                max_words=100,
                relative_scaling=0.5
            ).generate(processed_text)

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Get word frequencies
            word_frequencies = wordcloud.words_

            return {
                'wordcloud_created': True,
                'word_frequencies': dict(list(word_frequencies.items())[:20]),
                'total_words': len(processed_text.split()),
                'unique_words': len(word_frequencies),
                'esg_category': esg_category,
                'title': title
            }

        except Exception as e:
            return {'error': str(e)}

    def create_cooccurrence_network(self, documents: List[str], top_words: int = 20,
                                  min_cooccurrence: int = 2) -> Dict[str, Any]:
        """
        Create word co-occurrence network visualization.

        Parameters:
        -----------
        documents : List[str]
            List of documents
        top_words : int
            Number of top words to include
        min_cooccurrence : int
            Minimum co-occurrence threshold

        Returns:
        --------
        Dict with co-occurrence network data
        """
        try:
            if not documents:
                return {'error': 'No documents provided'}

            # Preprocess documents
            preprocessor = TextPreprocessor(self.config)
            processed_docs = [preprocessor.preprocess_text(doc) for doc in documents]
            processed_docs = [doc for doc in processed_docs if doc.strip()]

            if not processed_docs:
                return {'error': 'No valid documents after preprocessing'}

            # Create document-term matrix
            dtm_result = preprocessor.create_document_term_matrix(processed_docs, method='count')

            if 'error' in dtm_result:
                return dtm_result

            dtm_df = dtm_result['document_term_matrix']

            # Get top words
            word_counts = dtm_df.sum(axis=0).sort_values(ascending=False)
            top_words_list = word_counts.head(top_words).index.tolist()

            # Calculate co-occurrence matrix
            cooccurrence_matrix = self._calculate_cooccurrence_matrix(
                processed_docs, top_words_list, min_cooccurrence
            )

            # Create network graph
            network_data = self._create_network_graph(cooccurrence_matrix, min_cooccurrence)

            # Plot network
            self._plot_cooccurrence_network(network_data['graph'], top_words_list)

            return {
                'cooccurrence_matrix': cooccurrence_matrix,
                'network_graph': network_data['graph'],
                'top_words': top_words_list,
                'network_metrics': network_data['metrics'],
                'min_cooccurrence': min_cooccurrence,
                'num_documents': len(processed_docs)
            }

        except Exception as e:
            return {'error': str(e)}

    def create_esg_category_comparison(self, documents_by_category: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Create comparative visualization of ESG categories.

        Parameters:
        -----------
        documents_by_category : Dict[str, List[str]]
            Documents grouped by ESG category

        Returns:
        --------
        Dict with comparison analysis
        """
        try:
            if not documents_by_category:
                return {'error': 'No categorized documents provided'}

            category_analysis = {}
            preprocessor = TextPreprocessor(self.config)

            # Analyze each category
            for category, docs in documents_by_category.items():
                if not docs:
                    continue

                # Combine all documents in category
                combined_text = ' '.join(docs)
                processed_text = preprocessor.preprocess_text(combined_text)

                # Get word frequencies
                words = processed_text.split()
                word_freq = Counter(words)

                # Create document-term matrix for this category
                dtm_result = preprocessor.create_document_term_matrix(docs, method='tfidf')

                category_analysis[category] = {
                    'document_count': len(docs),
                    'total_words': len(words),
                    'unique_words': len(word_freq),
                    'top_words': dict(word_freq.most_common(20)),
                    'avg_doc_length': np.mean([len(doc.split()) for doc in docs]),
                    'dtm_features': dtm_result.get('feature_importance', {})
                }

            # Create comparative visualizations
            self._plot_category_comparison(category_analysis)

            return {
                'category_analysis': category_analysis,
                'categories': list(documents_by_category.keys()),
                'total_documents': sum(len(docs) for docs in documents_by_category.values()),
                'comparison_created': True
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_cooccurrence_matrix(self, documents: List[str], words: List[str],
                                     min_cooccurrence: int) -> pd.DataFrame:
        """Calculate word co-occurrence matrix."""
        try:
            # Initialize co-occurrence matrix
            cooc_matrix = pd.DataFrame(0, index=words, columns=words)

            # Count co-occurrences
            for doc in documents:
                doc_words = doc.split()
                doc_word_set = set(doc_words) & set(words)

                # Update co-occurrence counts
                for word1 in doc_word_set:
                    for word2 in doc_word_set:
                        if word1 != word2:
                            cooc_matrix.loc[word1, word2] += 1

            # Apply minimum threshold
            cooc_matrix[cooc_matrix < min_cooccurrence] = 0

            return cooc_matrix

        except Exception:
            return pd.DataFrame()

    def _create_network_graph(self, cooccurrence_matrix: pd.DataFrame,
                            min_cooccurrence: int) -> Dict[str, Any]:
        """Create network graph from co-occurrence matrix."""
        try:
            # Create NetworkX graph
            G = nx.Graph()

            # Add nodes
            for word in cooccurrence_matrix.index:
                G.add_node(word)

            # Add edges
            for word1 in cooccurrence_matrix.index:
                for word2 in cooccurrence_matrix.columns:
                    weight = cooccurrence_matrix.loc[word1, word2]
                    if weight >= min_cooccurrence and word1 != word2:
                        G.add_edge(word1, word2, weight=weight)

            # Calculate network metrics
            metrics = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
            }

            # Add centrality measures
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                metrics['top_central_words'] = dict(sorted(centrality.items(),
                                                         key=lambda x: x[1], reverse=True)[:10])

            return {
                'graph': G,
                'metrics': metrics
            }

        except Exception:
            return {'graph': nx.Graph(), 'metrics': {}}

    def _plot_cooccurrence_network(self, graph: nx.Graph, words: List[str]):
        """Plot co-occurrence network."""
        try:
            if graph.number_of_nodes() == 0:
                return

            plt.figure(figsize=self.config.figure_size)

            # Create layout
            pos = nx.spring_layout(graph, k=1, iterations=50)

            # Draw network
            nx.draw_networkx_nodes(graph, pos, node_color='lightblue',
                                 node_size=500, alpha=0.7)
            nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1)
            nx.draw_networkx_labels(graph, pos, font_size=8)

            plt.title("Word Co-occurrence Network", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()

        except Exception as e:
            logger.error(f"Error plotting network: {e}")

    def _plot_category_comparison(self, category_analysis: Dict[str, Any]):
        """Plot comparison between ESG categories."""
        try:
            categories = list(category_analysis.keys())
            if len(categories) < 2:
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Document counts
            doc_counts = [category_analysis[cat]['document_count'] for cat in categories]
            axes[0, 0].bar(categories, doc_counts, color=['green', 'blue', 'orange'][:len(categories)])
            axes[0, 0].set_title('Document Count by ESG Category')
            axes[0, 0].set_ylabel('Number of Documents')

            # Average document length
            avg_lengths = [category_analysis[cat]['avg_doc_length'] for cat in categories]
            axes[0, 1].bar(categories, avg_lengths, color=['green', 'blue', 'orange'][:len(categories)])
            axes[0, 1].set_title('Average Document Length by Category')
            axes[0, 1].set_ylabel('Average Words per Document')

            # Unique words
            unique_words = [category_analysis[cat]['unique_words'] for cat in categories]
            axes[1, 0].bar(categories, unique_words, color=['green', 'blue', 'orange'][:len(categories)])
            axes[1, 0].set_title('Vocabulary Size by Category')
            axes[1, 0].set_ylabel('Unique Words')

            # Word frequency comparison (top 10 words)
            axes[1, 1].axis('off')  # Use for text summary or leave empty

            plt.tight_layout()

        except Exception as e:
            logger.error(f"Error plotting category comparison: {e}")


class TopicModeling:
    """
    Topic modeling using LDA and other techniques.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.preprocessor = TextPreprocessor(config)

    def latent_dirichlet_allocation(self, documents: List[str],
                                  n_topics: int = None) -> Dict[str, Any]:
        """
        Perform topic modeling using Latent Dirichlet Allocation.

        Parameters:
        -----------
        documents : List[str]
            List of documents
        n_topics : int
            Number of topics to extract

        Returns:
        --------
        Dict with topic modeling results
        """
        try:
            if n_topics is None:
                n_topics = self.config.n_topics

            # Create document-term matrix
            dtm_result = self.preprocessor.create_document_term_matrix(documents, method='count')

            if 'error' in dtm_result:
                return dtm_result

            dtm = dtm_result['document_term_matrix']
            feature_names = dtm_result['feature_names']

            # Fit LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )

            lda.fit(dtm)

            # Extract topics
            topics = self._extract_topic_words(lda, feature_names, self.config.topic_words)

            # Get document-topic distributions
            doc_topic_dist = lda.transform(dtm)

            # Calculate topic coherence (simplified)
            topic_coherence = self._calculate_topic_coherence(topics, documents)

            # Assign dominant topic to each document
            dominant_topics = np.argmax(doc_topic_dist, axis=1)

            # Topic statistics
            topic_stats = self._calculate_topic_statistics(doc_topic_dist, documents)

            return {
                'topics': topics,
                'document_topic_distribution': doc_topic_dist,
                'dominant_topics': dominant_topics.tolist(),
                'topic_coherence': topic_coherence,
                'topic_statistics': topic_stats,
                'n_topics': n_topics,
                'model': lda,
                'feature_names': feature_names,
                'perplexity': lda.perplexity(dtm),
                'log_likelihood': lda.score(dtm)
            }

        except Exception as e:
            return {'error': str(e)}

    def esg_topic_modeling(self, documents: List[str],
                          target_esg_categories: List[str] = None) -> Dict[str, Any]:
        """
        Perform topic modeling focused on ESG themes.

        Parameters:
        -----------
        documents : List[str]
            List of documents
        target_esg_categories : List[str]
            Specific ESG categories to focus on

        Returns:
        --------
        Dict with ESG-focused topic modeling results
        """
        try:
            if target_esg_categories is None:
                target_esg_categories = ['environmental', 'social', 'governance']

            # Filter documents for ESG relevance
            esg_documents = []
            for doc in documents:
                doc_categories = self._identify_esg_categories_in_doc(doc)
                if any(cat in target_esg_categories for cat in doc_categories):
                    esg_documents.append(doc)

            if not esg_documents:
                return {'error': 'No ESG-relevant documents found'}

            # Perform LDA with ESG-specific preprocessing
            lda_result = self.latent_dirichlet_allocation(
                esg_documents,
                n_topics=len(target_esg_categories) * 2  # 2 subtopics per ESG category
            )

            if 'error' in lda_result:
                return lda_result

            # Map topics to ESG categories
            topic_esg_mapping = self._map_topics_to_esg(lda_result['topics'], target_esg_categories)

            # ESG topic analysis
            esg_analysis = self._analyze_esg_topics(
                lda_result, topic_esg_mapping, esg_documents
            )

            return {
                'esg_topics': lda_result['topics'],
                'topic_esg_mapping': topic_esg_mapping,
                'esg_analysis': esg_analysis,
                'esg_documents_count': len(esg_documents),
                'target_categories': target_esg_categories,
                'document_topic_distribution': lda_result['document_topic_distribution'],
                'esg_topic_coherence': lda_result['topic_coherence']
            }

        except Exception as e:
            return {'error': str(e)}

    def dynamic_topic_modeling(self, documents: List[str],
                             timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Perform dynamic topic modeling to track topic evolution over time.

        Parameters:
        -----------
        documents : List[str]
            List of documents
        timestamps : List[datetime]
            Timestamps for each document

        Returns:
        --------
        Dict with dynamic topic modeling results
        """
        try:
            if len(documents) != len(timestamps):
                return {'error': 'Documents and timestamps must have same length'}

            # Group documents by time periods (monthly)
            doc_groups = self._group_documents_by_time(documents, timestamps)

            if len(doc_groups) < 2:
                return {'error': 'Need at least 2 time periods for dynamic modeling'}

            # Perform LDA for each time period
            temporal_topics = {}
            all_topics = []

            for period, period_docs in doc_groups.items():
                if len(period_docs) >= 5:  # Minimum documents per period
                    lda_result = self.latent_dirichlet_allocation(period_docs)
                    if 'error' not in lda_result:
                        temporal_topics[period] = lda_result
                        all_topics.extend(lda_result['topics'])

            # Track topic evolution
            topic_evolution = self._track_topic_evolution(temporal_topics)

            # Identify trending topics
            trending_topics = self._identify_trending_topics(temporal_topics)

            return {
                'temporal_topics': temporal_topics,
                'topic_evolution': topic_evolution,
                'trending_topics': trending_topics,
                'time_periods': list(doc_groups.keys()),
                'documents_per_period': {period: len(docs) for period, docs in doc_groups.items()}
            }

        except Exception as e:
            return {'error': str(e)}

    def _extract_topic_words(self, lda_model, feature_names: List[str],
                           n_words: int) -> List[Dict[str, Any]]:
        """Extract top words for each topic."""
        try:
            topics = []

            for topic_idx, topic in enumerate(lda_model.components_):
                # Get top word indices
                top_word_indices = topic.argsort()[-n_words:][::-1]

                # Get words and weights
                topic_words = []
                for word_idx in top_word_indices:
                    word = feature_names[word_idx]
                    weight = topic[word_idx]
                    topic_words.append({'word': word, 'weight': round(weight, 4)})

                topics.append({
                    'topic_id': topic_idx,
                    'words': topic_words,
                    'top_words': [w['word'] for w in topic_words[:5]]
                })

            return topics

        except Exception:
            return []

    def _calculate_topic_coherence(self, topics: List[Dict[str, Any]],
                                 documents: List[str]) -> Dict[str, float]:
        """Calculate topic coherence scores (simplified implementation)."""
        try:
            coherence_scores = {}

            for topic in topics:
                topic_id = topic['topic_id']
                top_words = topic['top_words']

                # Calculate pairwise word co-occurrence
                cooccurrence_count = 0
                total_pairs = 0

                for i, word1 in enumerate(top_words):
                    for j, word2 in enumerate(top_words[i+1:], i+1):
                        total_pairs += 1

                        # Count documents containing both words
                        cooccurrence = sum(1 for doc in documents
                                         if word1.lower() in doc.lower() and word2.lower() in doc.lower())

                        if cooccurrence > 0:
                            cooccurrence_count += 1

                # Coherence as ratio of co-occurring pairs
                coherence = cooccurrence_count / total_pairs if total_pairs > 0 else 0
                coherence_scores[f'topic_{topic_id}'] = round(coherence, 4)

            return coherence_scores

        except Exception:
            return {}

    def _calculate_topic_statistics(self, doc_topic_dist: np.ndarray,
                                  documents: List[str]) -> Dict[str, Any]:
        """Calculate statistics about topic distribution."""
        try:
            n_topics = doc_topic_dist.shape[1]

            stats = {
                'topic_prevalence': {},
                'topic_concentration': {},
                'dominant_topic_distribution': {}
            }

            # Topic prevalence (average probability across documents)
            for topic_idx in range(n_topics):
                prevalence = np.mean(doc_topic_dist[:, topic_idx])
                stats['topic_prevalence'][f'topic_{topic_idx}'] = round(prevalence, 4)

            # Topic concentration (how concentrated topics are in documents)
            topic_entropy = -np.sum(doc_topic_dist * np.log(doc_topic_dist + 1e-10), axis=1)
            stats['avg_topic_entropy'] = round(np.mean(topic_entropy), 4)

            # Dominant topic distribution
            dominant_topics = np.argmax(doc_topic_dist, axis=1)
            for topic_idx in range(n_topics):
                count = np.sum(dominant_topics == topic_idx)
                stats['dominant_topic_distribution'][f'topic_{topic_idx}'] = int(count)

            return stats

        except Exception:
            return {}

    def _identify_esg_categories_in_doc(self, document: str) -> List[str]:
        """Identify ESG categories mentioned in a document."""
        categories = []
        doc_lower = document.lower()

        for category, keywords in self.config.esg_keywords.items():
            if any(keyword.lower() in doc_lower for keyword in keywords):
                categories.append(category)

        return categories

    def _map_topics_to_esg(self, topics: List[Dict[str, Any]],
                          esg_categories: List[str]) -> Dict[str, str]:
        """Map topics to ESG categories based on keyword overlap."""
        try:
            topic_mapping = {}

            for topic in topics:
                topic_id = topic['topic_id']
                topic_words = [w['word'] for w in topic['words']]

                # Calculate overlap with each ESG category
                best_category = None
                best_overlap = 0

                for category in esg_categories:
                    esg_keywords = self.config.esg_keywords.get(category, [])
                    overlap = sum(1 for word in topic_words
                                if any(keyword.lower() in word.lower() for keyword in esg_keywords))

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_category = category

                topic_mapping[f'topic_{topic_id}'] = best_category or 'other'

            return topic_mapping

        except Exception:
            return {}

    def _analyze_esg_topics(self, lda_result: Dict[str, Any],
                          topic_mapping: Dict[str, str],
                          documents: List[str]) -> Dict[str, Any]:
        """Analyze ESG-specific aspects of topics."""
        try:
            esg_analysis = {}

            # Group topics by ESG category
            for category in ['environmental', 'social', 'governance']:
                category_topics = [topic_id for topic_id, mapped_category in topic_mapping.items()
                                 if mapped_category == category]

                if category_topics:
                    # Calculate category prevalence
                    topic_indices = [int(tid.split('_')[1]) for tid in category_topics]
                    doc_topic_dist = lda_result['document_topic_distribution']
                    category_prevalence = np.mean(doc_topic_dist[:, topic_indices])

                    esg_analysis[category] = {
                        'topics': category_topics,
                        'prevalence': round(category_prevalence, 4),
                        'topic_count': len(category_topics)
                    }

            return esg_analysis

        except Exception:
            return {}

    def _group_documents_by_time(self, documents: List[str],
                               timestamps: List[datetime]) -> Dict[str, List[str]]:
        """Group documents by time periods."""
        try:
            doc_groups = defaultdict(list)

            for doc, timestamp in zip(documents, timestamps):
                # Group by year-month
                period = timestamp.strftime("%Y-%m")
                doc_groups[period].append(doc)

            return dict(doc_groups)

        except Exception:
            return {}

    def _track_topic_evolution(self, temporal_topics: Dict[str, Any]) -> Dict[str, Any]:
        """Track how topics evolve over time."""
        try:
            evolution = {}

            # This is a simplified implementation
            # In practice, you'd use more sophisticated topic alignment methods

            periods = sorted(temporal_topics.keys())

            for i, period in enumerate(periods):
                if period in temporal_topics:
                    topics = temporal_topics[period]['topics']
                    evolution[period] = {
                        'topic_count': len(topics),
                        'top_words_per_topic': [topic['top_words'][:3] for topic in topics]
                    }

            return evolution

        except Exception:
            return {}

    def _identify_trending_topics(self, temporal_topics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify topics that are trending over time."""
        try:
            # Simplified trending analysis
            trending = []

            periods = sorted(temporal_topics.keys())
            if len(periods) >= 2:
                recent_topics = temporal_topics[periods[-1]]['topics']
                trending = [{'topic_words': topic['top_words'][:5], 'trend': 'emerging'}
                           for topic in recent_topics[:3]]

            return trending

        except Exception:
            return []


class SentimentAnalyzer:
    """
    Multi-dimensional sentiment analysis for ESG content.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.vader_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

    def analyze_sentiment(self, text: str, method: str = 'combined') -> Dict[str, Any]:
        """
        Analyze sentiment using multiple methods.

        Parameters:
        -----------
        text : str
            Text to analyze
        method : str
            Analysis method ('textblob', 'vader', 'combined')

        Returns:
        --------
        Dict with sentiment analysis results
        """
        try:
            if not text or not isinstance(text, str):
                return {'error': 'Invalid text input'}

            results = {}

            # TextBlob analysis
            if method in ['textblob', 'combined']:
                blob = TextBlob(text)
                results['textblob'] = {
                    'polarity': round(blob.sentiment.polarity, self.config.precision),
                    'subjectivity': round(blob.sentiment.subjectivity, self.config.precision)
                }

            # VADER analysis
            if method in ['vader', 'combined'] and self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                results['vader'] = {
                    'compound': round(vader_scores['compound'], self.config.precision),
                    'positive': round(vader_scores['pos'], self.config.precision),
                    'negative': round(vader_scores['neg'], self.config.precision),
                    'neutral': round(vader_scores['neu'], self.config.precision)
                }

            # Combined analysis
            if method == 'combined' and 'textblob' in results:
                if 'vader' in results:
                    # Average the sentiment scores
                    combined_sentiment = (results['textblob']['polarity'] +
                                        results['vader']['compound']) / 2
                else:
                    combined_sentiment = results['textblob']['polarity']

                results['combined'] = {
                    'sentiment': round(combined_sentiment, self.config.precision),
                    'classification': self._classify_sentiment(combined_sentiment),
                    'confidence': self._calculate_confidence(results)
                }

            # Text statistics
            results['text_stats'] = {
                'length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(sent_tokenize(text))
            }

            return results

        except Exception as e:
            return {'error': str(e)}

    def esg_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment specifically for ESG themes.

        Parameters:
        -----------
        text : str
            Text to analyze for ESG sentiment

        Returns:
        --------
        Dict with ESG-specific sentiment analysis
        """
        try:
            # Overall sentiment
            overall_sentiment = self.analyze_sentiment(text, method='combined')

            if 'error' in overall_sentiment:
                return overall_sentiment

            # ESG category-specific sentiment
            esg_sentiments = {}

            for category, keywords in self.config.esg_keywords.items():
                # Extract sentences mentioning ESG keywords
                category_sentences = self._extract_category_sentences(text, keywords)

                if category_sentences:
                    category_text = ' '.join(category_sentences)
                    category_sentiment = self.analyze_sentiment(category_text, method='combined')

                    esg_sentiments[category] = {
                        'sentiment_score': category_sentiment.get('combined', {}).get('sentiment', 0),
                        'classification': category_sentiment.get('combined', {}).get('classification', 'neutral'),
                        'sentence_count': len(category_sentences),
                        'keyword_mentions': self._count_keyword_mentions(category_text, keywords)
                    }
                else:
                    esg_sentiments[category] = {
                        'sentiment_score': 0,
                        'classification': 'not_mentioned',
                        'sentence_count': 0,
                        'keyword_mentions': 0
                    }

            # Calculate ESG sentiment scores
            esg_scores = self._calculate_esg_scores(esg_sentiments)

            return {
                'overall_sentiment': overall_sentiment,
                'esg_category_sentiments': esg_sentiments,
                'esg_scores': esg_scores,
                'esg_summary': self._summarize_esg_sentiment(esg_sentiments, esg_scores)
            }

        except Exception as e:
            return {'error': str(e)}

    def sentiment_time_series(self, texts: List[str], timestamps: List[datetime],
                            window_size: int = 7) -> Dict[str, Any]:
        """
        Analyze sentiment over time with moving averages.

        Parameters:
        -----------
        texts : List[str]
            List of texts with timestamps
        timestamps : List[datetime]
            Corresponding timestamps
        window_size : int
            Moving average window size

        Returns:
        --------
        Dict with time series sentiment analysis
        """
        try:
            if len(texts) != len(timestamps):
                return {'error': 'Texts and timestamps must have same length'}

            # Sort by timestamp
            sorted_data = sorted(zip(timestamps, texts))
            sorted_timestamps, sorted_texts = zip(*sorted_data)

            # Calculate sentiment for each text
            sentiment_scores = []
            esg_scores = []

            for text in sorted_texts:
                sentiment_result = self.analyze_sentiment(text, method='combined')
                esg_result = self.esg_sentiment_analysis(text)

                sentiment_score = sentiment_result.get('combined', {}).get('sentiment', 0)
                sentiment_scores.append(sentiment_score)

                # Average ESG scores
                if 'esg_scores' in esg_result:
                    avg_esg_score = np.mean(list(esg_result['esg_scores'].values()))
                else:
                    avg_esg_score = 0
                esg_scores.append(avg_esg_score)

            # Calculate moving averages
            sentiment_ma = self._calculate_moving_average(sentiment_scores, window_size)
            esg_ma = self._calculate_moving_average(esg_scores, window_size)

            # Trend analysis
            sentiment_trend = self._analyze_trend(sentiment_scores)
            esg_trend = self._analyze_trend(esg_scores)

            # Create time series DataFrame
            ts_data = pd.DataFrame({
                'timestamp': sorted_timestamps,
                'sentiment_score': sentiment_scores,
                'esg_score': esg_scores,
                'sentiment_ma': sentiment_ma,
                'esg_ma': esg_ma
            })

            return {
                'time_series_data': ts_data,
                'sentiment_trend': sentiment_trend,
                'esg_trend': esg_trend,
                'summary_stats': {
                    'avg_sentiment': round(np.mean(sentiment_scores), self.config.precision),
                    'avg_esg_score': round(np.mean(esg_scores), self.config.precision),
                    'sentiment_volatility': round(np.std(sentiment_scores), self.config.precision),
                    'esg_volatility': round(np.std(esg_scores), self.config.precision)
                },
                'window_size': window_size
            }

        except Exception as e:
            return {'error': str(e)}

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into categories."""
        if score > self.config.sentiment_threshold:
            return 'positive'
        elif score < -self.config.sentiment_threshold:
            return 'negative'
        else:
            return 'neutral'

    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in sentiment analysis."""
        try:
            if 'textblob' in results and 'vader' in results:
                # If both methods agree, confidence is higher
                tb_sentiment = results['textblob']['polarity']
                vader_sentiment = results['vader']['compound']

                agreement = 1 - abs(tb_sentiment - vader_sentiment) / 2
                return round(agreement, self.config.precision)
            else:
                return 0.5  # Default moderate confidence
        except Exception:
            return 0.5

    def _extract_category_sentences(self, text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing ESG category keywords."""
        try:
            sentences = sent_tokenize(text)
            category_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword.lower() in sentence_lower for keyword in keywords):
                    category_sentences.append(sentence)

            return category_sentences

        except Exception:
            return []

    def _count_keyword_mentions(self, text: str, keywords: List[str]) -> int:
        """Count total keyword mentions in text."""
        try:
            text_lower = text.lower()
            count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
            return count
        except Exception:
            return 0

    def _calculate_esg_scores(self, esg_sentiments: Dict[str, Any]) -> Dict[str, float]:
        """Calculate normalized ESG scores."""
        try:
            scores = {}

            for category, sentiment_data in esg_sentiments.items():
                if sentiment_data['classification'] != 'not_mentioned':
                    # Normalize sentiment score to 0-100 scale
                    raw_score = sentiment_data['sentiment_score']
                    normalized_score = (raw_score + 1) * 50  # Convert from [-1,1] to [0,100]
                    scores[category] = round(normalized_score, self.config.precision)
                else:
                    scores[category] = 50  # Neutral score for not mentioned

            return scores

        except Exception:
            return {}

    def _summarize_esg_sentiment(self, esg_sentiments: Dict[str, Any],
                               esg_scores: Dict[str, float]) -> Dict[str, Any]:
        """Create summary of ESG sentiment analysis."""
        try:
            # Overall ESG sentiment
            mentioned_categories = [cat for cat, data in esg_sentiments.items()
                                  if data['classification'] != 'not_mentioned']

            if mentioned_categories:
                avg_score = np.mean([esg_scores[cat] for cat in mentioned_categories])
                overall_classification = self._classify_sentiment((avg_score - 50) / 50)  # Convert back to [-1,1]
            else:
                avg_score = 50
                overall_classification = 'neutral'

            # Best and worst performing categories
            if esg_scores:
                best_category = max(esg_scores, key=esg_scores.get)
                worst_category = min(esg_scores, key=esg_scores.get)
            else:
                best_category = None
                worst_category = None

            return {
                'overall_esg_score': round(avg_score, self.config.precision),
                'overall_classification': overall_classification,
                'categories_mentioned': mentioned_categories,
                'best_category': best_category,
                'worst_category': worst_category,
                'category_coverage': len(mentioned_categories) / len(esg_sentiments)
            }

        except Exception:
            return {}

    def _calculate_moving_average(self, values: List[float], window_size: int) -> List[float]:
        """Calculate moving average of values."""
        try:
            if len(values) < window_size:
                return values[:]

            moving_avg = []
            for i in range(len(values)):
                if i < window_size - 1:
                    # Use expanding window for initial values
                    avg = np.mean(values[:i+1])
                else:
                    # Use rolling window
                    avg = np.mean(values[i-window_size+1:i+1])
                moving_avg.append(avg)

            return moving_avg

        except Exception:
            return values[:]

    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in time series values."""
        try:
            if len(values) < 2:
                return {'trend': 'insufficient_data', 'slope': 0}

            # Linear regression to find trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

            # Classify trend
            if abs(slope) < 0.001:
                trend = 'stable'
            elif slope > 0:
                trend = 'improving'
            else:
                trend = 'declining'

            return {
                'trend': trend,
                'slope': round(slope, self.config.precision),
                'r_squared': round(r_value ** 2, self.config.precision),
                'p_value': round(p_value, self.config.precision)
            }

        except Exception:
            return {'trend': 'unknown', 'slope': 0}


class SentimentSignalGenerator:
    """
    Generate trading/investment signals from sentiment analysis.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.sentiment_analyzer = SentimentAnalyzer(config)

    def generate_esg_sentiment_signal(self, company_texts: Dict[str, List[str]],
                                    timestamps: Dict[str, List[datetime]] = None) -> Dict[str, Any]:
        """
        Generate ESG sentiment signals for companies.

        Parameters:
        -----------
        company_texts : Dict[str, List[str]]
            Texts for each company
        timestamps : Dict[str, List[datetime]]
            Timestamps for each company's texts

        Returns:
        --------
        Dict with sentiment signals
        """
        try:
            company_signals = {}

            for company, texts in company_texts.items():
                if not texts:
                    continue

                # Analyze sentiment for all texts
                company_sentiments = []
                esg_scores = []

                for text in texts:
                    sentiment_result = self.sentiment_analyzer.esg_sentiment_analysis(text)

                    if 'error' not in sentiment_result:
                        overall_sentiment = sentiment_result.get('overall_sentiment', {}).get('combined', {}).get('sentiment', 0)
                        company_sentiments.append(overall_sentiment)

                        # Extract ESG scores
                        esg_score_data = sentiment_result.get('esg_scores', {})
                        avg_esg = np.mean(list(esg_score_data.values())) if esg_score_data else 50
                        esg_scores.append(avg_esg)

                if company_sentiments:
                    # Calculate signal metrics
                    signal_data = self._calculate_signal_metrics(
                        company_sentiments, esg_scores, texts,
                        timestamps.get(company) if timestamps else None
                    )

                    company_signals[company] = signal_data

            # Generate portfolio-level signals
            portfolio_signal = self._generate_portfolio_signal(company_signals)

            return {
                'company_signals': company_signals,
                'portfolio_signal': portfolio_signal,
                'signal_date': datetime.now(),
                'companies_analyzed': list(company_texts.keys())
            }

        except Exception as e:
            return {'error': str(e)}

    def create_sentiment_momentum_signal(self, sentiment_scores: List[float],
                                       timestamps: List[datetime],
                                       lookback_periods: int = 10) -> Dict[str, Any]:
        """
        Create momentum-based sentiment signal.

        Parameters:
        -----------
        sentiment_scores : List[float]
            Historical sentiment scores
        timestamps : List[datetime]
            Corresponding timestamps
        lookback_periods : int
            Number of periods for momentum calculation

        Returns:
        --------
        Dict with momentum signal
        """
        try:
            if len(sentiment_scores) < lookback_periods:
                return {'error': f'Need at least {lookback_periods} data points'}

            # Calculate momentum
            momentum_scores = []

            for i in range(lookback_periods, len(sentiment_scores)):
                current_avg = np.mean(sentiment_scores[i-lookback_periods//2:i])
                previous_avg = np.mean(sentiment_scores[i-lookback_periods:i-lookback_periods//2])

                momentum = current_avg - previous_avg
                momentum_scores.append(momentum)

            # Generate signals
            signal_timestamps = timestamps[lookback_periods:]
            signals = []

            for momentum in momentum_scores:
                if momentum > 0.1:
                    signal = 'buy'
                elif momentum < -0.1:
                    signal = 'sell'
                else:
                    signal = 'hold'
                signals.append(signal)

            # Calculate signal statistics
            signal_stats = {
                'buy_signals': signals.count('buy'),
                'sell_signals': signals.count('sell'),
                'hold_signals': signals.count('hold'),
                'avg_momentum': round(np.mean(momentum_scores), self.config.precision),
                'momentum_volatility': round(np.std(momentum_scores), self.config.precision)
            }

            return {
                'momentum_scores': [round(m, self.config.precision) for m in momentum_scores],
                'signals': signals,
                'signal_timestamps': signal_timestamps,
                'signal_statistics': signal_stats,
                'lookback_periods': lookback_periods
            }

        except Exception as e:
            return {'error': str(e)}

    def sentiment_divergence_signal(self, sentiment_scores: List[float],
                                  price_data: List[float],
                                  timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Detect sentiment-price divergence signals.

        Parameters:
        -----------
        sentiment_scores : List[float]
            Sentiment scores
        price_data : List[float]
            Corresponding price data
        timestamps : List[datetime]
            Timestamps

        Returns:
        --------
        Dict with divergence signals
        """
        try:
            if len(sentiment_scores) != len(price_data) or len(sentiment_scores) != len(timestamps):
                return {'error': 'All input lists must have same length'}

            if len(sentiment_scores) < 20:
                return {'error': 'Need at least 20 data points for divergence analysis'}

            # Calculate trends
            window = 10
            sentiment_trends = []
            price_trends = []
            divergence_signals = []

            for i in range(window, len(sentiment_scores)):
                # Sentiment trend
                sentiment_window = sentiment_scores[i-window:i]
                sentiment_slope = self._calculate_trend_slope(sentiment_window)
                sentiment_trends.append(sentiment_slope)

                # Price trend
                price_window = price_data[i-window:i]
                price_slope = self._calculate_trend_slope(price_window)
                price_trends.append(price_slope)

                # Detect divergence
                if sentiment_slope > 0.01 and price_slope < -0.01:
                    # Positive sentiment, negative price trend - potential buy
                    signal = 'bullish_divergence'
                elif sentiment_slope < -0.01 and price_slope > 0.01:
                    # Negative sentiment, positive price trend - potential sell
                    signal = 'bearish_divergence'
                else:
                    signal = 'no_divergence'

                divergence_signals.append(signal)

            # Signal statistics
            signal_stats = {
                'bullish_divergences': divergence_signals.count('bullish_divergence'),
                'bearish_divergences': divergence_signals.count('bearish_divergence'),
                'no_divergence_periods': divergence_signals.count('no_divergence'),
                'sentiment_price_correlation': round(np.corrcoef(sentiment_scores, price_data)[0, 1], self.config.precision)
            }

            return {
                'divergence_signals': divergence_signals,
                'sentiment_trends': [round(t, self.config.precision) for t in sentiment_trends],
                'price_trends': [round(t, self.config.precision) for t in price_trends],
                'signal_timestamps': timestamps[window:],
                'signal_statistics': signal_stats,
                'window_size': window
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_signal_metrics(self, sentiments: List[float], esg_scores: List[float],
                                texts: List[str], timestamps: List[datetime] = None) -> Dict[str, Any]:
        """Calculate comprehensive signal metrics for a company."""
        try:
            # Basic statistics
            avg_sentiment = np.mean(sentiments)
            sentiment_volatility = np.std(sentiments)
            avg_esg_score = np.mean(esg_scores)

            # Trend analysis
            sentiment_trend = self._calculate_trend_slope(sentiments)
            esg_trend = self._calculate_trend_slope(esg_scores)

            # Signal strength
            signal_strength = self._calculate_signal_strength(avg_sentiment, sentiment_volatility, sentiment_trend)

            # Generate primary signal
            primary_signal = self._generate_primary_signal(avg_sentiment, sentiment_trend, signal_strength)

            # ESG signal
            esg_signal = self._generate_esg_signal(avg_esg_score, esg_trend)

            # Text volume signal
            volume_signal = self._generate_volume_signal(len(texts), timestamps)

            return {
                'average_sentiment': round(avg_sentiment, self.config.precision),
                'sentiment_volatility': round(sentiment_volatility, self.config.precision),
                'average_esg_score': round(avg_esg_score, self.config.precision),
                'sentiment_trend': round(sentiment_trend, self.config.precision),
                'esg_trend': round(esg_trend, self.config.precision),
                'signal_strength': round(signal_strength, self.config.precision),
                'primary_signal': primary_signal,
                'esg_signal': esg_signal,
                'volume_signal': volume_signal,
                'text_count': len(texts)
            }

        except Exception:
            return {}

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        try:
            if len(values) < 2:
                return 0

            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope

        except Exception:
            return 0

    def _calculate_signal_strength(self, avg_sentiment: float, volatility: float, trend: float) -> float:
        """Calculate overall signal strength."""
        try:
            # Combine magnitude, consistency (low volatility), and trend
            magnitude_score = abs(avg_sentiment)
            consistency_score = 1 / (1 + volatility)  # Lower volatility = higher consistency
            trend_score = abs(trend)

            # Weighted combination
            signal_strength = (magnitude_score * 0.4 + consistency_score * 0.3 + trend_score * 0.3)
            return min(signal_strength, 1.0)  # Cap at 1.0

        except Exception:
            return 0

    def _generate_primary_signal(self, avg_sentiment: float, trend: float, strength: float) -> Dict[str, Any]:
        """Generate primary investment signal."""
        try:
            # Signal logic
            if avg_sentiment > 0.2 and trend > 0.01 and strength > 0.6:
                signal = 'strong_buy'
            elif avg_sentiment > 0.1 and (trend > 0 or strength > 0.5):
                signal = 'buy'
            elif avg_sentiment < -0.2 and trend < -0.01 and strength > 0.6:
                signal = 'strong_sell'
            elif avg_sentiment < -0.1 and (trend < 0 or strength > 0.5):
                signal = 'sell'
            else:
                signal = 'hold'

            # Confidence based on strength and consistency
            if strength > 0.7:
                confidence = 'high'
            elif strength > 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'

            return {
                'signal': signal,
                'confidence': confidence,
                'strength': round(strength, self.config.precision)
            }

        except Exception:
            return {'signal': 'hold', 'confidence': 'low', 'strength': 0}

    def _generate_esg_signal(self, avg_esg_score: float, esg_trend: float) -> Dict[str, Any]:
        """Generate ESG-specific signal."""
        try:
            # ESG score is on 0-100 scale
            if avg_esg_score > 70 and esg_trend > 0:
                esg_signal = 'esg_positive'
            elif avg_esg_score > 60:
                esg_signal = 'esg_neutral_positive'
            elif avg_esg_score < 40 and esg_trend < 0:
                esg_signal = 'esg_negative'
            elif avg_esg_score < 50:
                esg_signal = 'esg_neutral_negative'
            else:
                esg_signal = 'esg_neutral'

            return {
                'signal': esg_signal,
                'score': round(avg_esg_score, self.config.precision),
                'trend': round(esg_trend, self.config.precision)
            }

        except Exception:
            return {'signal': 'esg_neutral', 'score': 50, 'trend': 0}

    def _generate_volume_signal(self, text_count: int, timestamps: List[datetime] = None) -> Dict[str, Any]:
        """Generate signal based on text volume and frequency."""
        try:
            # Volume-based signal
            if text_count > 50:
                volume_signal = 'high_attention'
            elif text_count > 20:
                volume_signal = 'medium_attention'
            elif text_count > 5:
                volume_signal = 'low_attention'
            else:
                volume_signal = 'minimal_attention'

            # Frequency analysis if timestamps available
            frequency_signal = 'unknown'
            if timestamps and len(timestamps) > 1:
                time_span = (max(timestamps) - min(timestamps)).days
                frequency = text_count / max(time_span, 1)  # texts per day

                if frequency > 2:
                    frequency_signal = 'high_frequency'
                elif frequency > 0.5:
                    frequency_signal = 'medium_frequency'
                else:
                    frequency_signal = 'low_frequency'

            return {
                'volume_signal': volume_signal,
                'frequency_signal': frequency_signal,
                'text_count': text_count
            }

        except Exception:
            return {'volume_signal': 'minimal_attention', 'frequency_signal': 'unknown', 'text_count': text_count}

    def _generate_portfolio_signal(self, company_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall portfolio signal from individual company signals."""
        try:
            if not company_signals:
                return {'signal': 'hold', 'confidence': 'low'}

            # Aggregate signals
            signal_counts = {'buy': 0, 'strong_buy': 0, 'sell': 0, 'strong_sell': 0, 'hold': 0}
            total_strength = 0

            for company, signals in company_signals.items():
                primary_signal = signals.get('primary_signal', {})
                signal = primary_signal.get('signal', 'hold')
                strength = primary_signal.get('strength', 0)

                if signal in signal_counts:
                    signal_counts[signal] += 1
                total_strength += strength

            # Generate portfolio signal
            total_companies = len(company_signals)
            avg_strength = total_strength / total_companies

            buy_ratio = (signal_counts['buy'] + signal_counts['strong_buy']) / total_companies
            sell_ratio = (signal_counts['sell'] + signal_counts['strong_sell']) / total_companies

            if buy_ratio > 0.6:
                portfolio_signal = 'portfolio_buy'
            elif sell_ratio > 0.6:
                portfolio_signal = 'portfolio_sell'
            elif buy_ratio > sell_ratio and buy_ratio > 0.4:
                portfolio_signal = 'portfolio_lean_buy'
            elif sell_ratio > buy_ratio and sell_ratio > 0.4:
                portfolio_signal = 'portfolio_lean_sell'
            else:
                portfolio_signal = 'portfolio_hold'

            return {
                'signal': portfolio_signal,
                'average_strength': round(avg_strength, self.config.precision),
                'buy_ratio': round(buy_ratio, self.config.precision),
                'sell_ratio': round(sell_ratio, self.config.precision),
                'signal_distribution': signal_counts
            }

        except Exception:
            return {'signal': 'portfolio_hold', 'confidence': 'low'}


class NamedEntityRecognizer:
    """
    Named Entity Recognition for ESG-relevant entities.
    """

    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.nlp = nlp_spacy

    def extract_entities(self, text: str, entity_types: List[str] = None) -> Dict[str, Any]:
        """
        Extract named entities from text.

        Parameters:
        -----------
        text : str
            Text to analyze
        entity_types : List[str]
            Specific entity types to extract

        Returns:
        --------
        Dict with extracted entities
        """
        try:
            if not text or not isinstance(text, str):
                return {'error': 'Invalid text input'}

            entities = {
                'companies': [],
                'people': [],
                'locations': [],
                'organizations': [],
                'money': [],
                'dates': [],
                'other': []
            }

            # Use spaCy if available
            if self.nlp:
                doc = self.nlp(text)

                for ent in doc.ents:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'confidence', 1.0)
                    }

                    # Categorize entities
                    if ent.label_ in ['ORG', 'CORP']:
                        entities['companies'].append(entity_info)
                    elif ent.label_ in ['PERSON']:
                        entities['people'].append(entity_info)
                    elif ent.label_ in ['GPE', 'LOC']:
                        entities['locations'].append(entity_info)
                    elif ent.label_ in ['ORG']:
                        entities['organizations'].append(entity_info)
                    elif ent.label_ in ['MONEY']:
                        entities['money'].append(entity_info)
                    elif ent.label_ in ['DATE', 'TIME']:
                        entities['dates'].append(entity_info)
                    else:
                        entities['other'].append(entity_info)

            else:
                # Fallback to NLTK
                entities = self._extract_entities_nltk(text)

            # Filter by requested entity types
            if entity_types:
                filtered_entities = {k: v for k, v in entities.items() if k in entity_types}
                entities = filtered_entities

            # Add entity statistics
            entity_stats = self._calculate_entity_statistics(entities)

            return {
                'entities': entities,
                'entity_statistics': entity_stats,
                'total_entities': sum(len(ents) for ents in entities.values())
            }

        except Exception as e:
            return {'error': str(e)}

    def extract_esg_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities specifically relevant to ESG analysis.

        Parameters:
        -----------
        text : str
            Text to analyze for ESG entities

        Returns:
        --------
        Dict with ESG-relevant entities
        """
        try:
            # General entity extraction
            entity_result = self.extract_entities(text)

            if 'error' in entity_result:
                return entity_result

            # ESG-specific entity extraction
            esg_entities = {
                'environmental_terms': [],
                'social_terms': [],
                'governance_terms': [],
                'sustainability_metrics': [],
                'regulatory_bodies': [],
                'esg_ratings_agencies': []
            }

            # Extract ESG-specific terms
            esg_entities.update(self._extract_esg_specific_entities(text))

            # Combine with general entities
            result = {
                'general_entities': entity_result['entities'],
                'esg_specific_entities': esg_entities,
                'entity_statistics': entity_result['entity_statistics'],
                'esg_entity_count': sum(len(entities) for entities in esg_entities.values())
            }

            # ESG entity analysis
            esg_analysis = self._analyze_esg_entities(esg_entities, text)
            result['esg_analysis'] = esg_analysis

            return result

        except Exception as e:
            return {'error': str(e)}

    def entity_relationship_extraction(self, text: str) -> Dict[str, Any]:
        """
        Extract relationships between entities.

        Parameters:
        -----------
        text : str
            Text to analyze for entity relationships

        Returns:
        --------
        Dict with entity relationships
        """
        try:
            if not self.nlp:
                return {'error': 'spaCy not available for relationship extraction'}

            doc = self.nlp(text)

            relationships = []
            entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]

            # Simple relationship extraction based on sentence structure
            for sent in doc.sents:
                sent_entities = []
                for ent in doc.ents:
                    if sent.start <= ent.start < sent.end:
                        sent_entities.append((ent.text, ent.label_))

                # If sentence has multiple entities, look for relationships
                if len(sent_entities) >= 2:
                    # Simple pattern: find verbs between entities
                    for token in sent:
                        if token.pos_ == 'VERB':
                            relationship = {
                                'sentence': sent.text,
                                'verb': token.text,
                                'entities': sent_entities,
                                'relationship_type': self._classify_relationship(token.text)
                            }
                            relationships.append(relationship)
                            break  # Take first verb for simplicity

            # Entity co-occurrence analysis
            cooccurrence = self._analyze_entity_cooccurrence(entities, text)

            return {
                'relationships': relationships,
                'entity_cooccurrence': cooccurrence,
                'total_relationships': len(relationships),
                'entities_found': len(entities)
            }

        except Exception as e:
            return {'error': str(e)}

    def track_entity_sentiment(self, texts: List[str], target_entities: List[str]) -> Dict[str, Any]:
        """
        Track sentiment associated with specific entities across texts.

        Parameters:
        -----------
        texts : List[str]
            List of texts to analyze
        target_entities : List[str]
            Entities to track sentiment for

        Returns:
        --------
        Dict with entity sentiment tracking
        """
        try:
            entity_sentiments = {entity: [] for entity in target_entities}
            sentiment_analyzer = SentimentAnalyzer(self.config)

            for text in texts:
                # Extract entities from text
                entity_result = self.extract_entities(text)

                if 'error' in entity_result:
                    continue

                # Find sentences containing target entities
                sentences = sent_tokenize(text)

                for entity in target_entities:
                    entity_sentences = []

                    for sentence in sentences:
                        if entity.lower() in sentence.lower():
                            entity_sentences.append(sentence)

                    if entity_sentences:
                        # Analyze sentiment of sentences mentioning the entity
                        combined_text = ' '.join(entity_sentences)
                        sentiment_result = sentiment_analyzer.analyze_sentiment(combined_text)

                        if 'combined' in sentiment_result:
                            sentiment_score = sentiment_result['combined']['sentiment']
                            entity_sentiments[entity].append({
                                'sentiment': sentiment_score,
                                'text_snippet': combined_text[:200] + '...' if len(combined_text) > 200 else combined_text,
                                'sentence_count': len(entity_sentences)
                            })

            # Calculate summary statistics for each entity
            entity_summary = {}
            for entity, sentiments in entity_sentiments.items():
                if sentiments:
                    scores = [s['sentiment'] for s in sentiments]
                    entity_summary[entity] = {
                        'average_sentiment': round(np.mean(scores), self.config.precision),
                        'sentiment_volatility': round(np.std(scores), self.config.precision),
                        'mention_count': len(sentiments),
                        'sentiment_trend': self._calculate_entity_sentiment_trend(scores)
                    }
                else:
                    entity_summary[entity] = {
                        'average_sentiment': 0,
                        'sentiment_volatility': 0,
                        'mention_count': 0,
                        'sentiment_trend': 'no_data'
                    }

            return {
                'entity_sentiments': entity_sentiments,
                'entity_summary': entity_summary,
                'target_entities': target_entities,
                'texts_analyzed': len(texts)
            }

        except Exception as e:
            return {'error': str(e)}

    def _extract_entities_nltk(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using NLTK as fallback."""
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            # Named entity chunking
            ne_tree = ne_chunk(pos_tags)

            entities = {
                'companies': [],
                'people': [],
                'locations': [],
                'organizations': [],
                'money': [],
                'dates': [],
                'other': []
            }

            # Extract entities from the tree
            for chunk in ne_tree:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    entity_info = {
                        'text': entity_text,
                        'label': chunk.label(),
                        'confidence': 0.8  # Default confidence for NLTK
                    }

                    # Categorize based on NLTK labels
                    if chunk.label() == 'PERSON':
                        entities['people'].append(entity_info)
                    elif chunk.label() == 'GPE':
                        entities['locations'].append(entity_info)
                    elif chunk.label() == 'ORGANIZATION':
                        entities['organizations'].append(entity_info)
                    else:
                        entities['other'].append(entity_info)

            return entities

        except Exception:
            return {
                'companies': [], 'people': [], 'locations': [],
                'organizations': [], 'money': [], 'dates': [], 'other': []
            }

    def _extract_esg_specific_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract ESG-specific entities and terms."""
        try:
            esg_entities = {
                'environmental_terms': [],
                'social_terms': [],
                'governance_terms': [],
                'sustainability_metrics': [],
                'regulatory_bodies': [],
                'esg_ratings_agencies': []
            }

            # Define ESG-specific patterns
            esg_patterns = {
                'environmental_terms': [
                    'carbon footprint', 'greenhouse gas', 'renewable energy', 'climate change',
                    'sustainability', 'biodiversity', 'water usage', 'waste management'
                ],
                'social_terms': [
                    'diversity and inclusion', 'human rights', 'labor practices', 'community engagement',
                    'employee welfare', 'workplace safety', 'social responsibility'
                ],
                'governance_terms': [
                    'board independence', 'executive compensation', 'audit committee', 'transparency',
                    'ethics and compliance', 'shareholder rights', 'risk management'
                ],
                'sustainability_metrics': [
                    'scope 1 emissions', 'scope 2 emissions', 'scope 3 emissions', 'esg score',
                    'sustainability index', 'carbon intensity', 'water intensity'
                ],
                'regulatory_bodies': [
                    'SEC', 'EPA', 'SASB', 'GRI', 'TCFD', 'UN Global Compact'
                ],
                'esg_ratings_agencies': [
                    'MSCI', 'Sustainalytics', 'ISS ESG', 'RepRisk', 'CDP', 'Refinitiv'
                ]
            }

            text_lower = text.lower()

            for category, patterns in esg_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        # Find all occurrences
                        start = 0
                        while True:
                            pos = text_lower.find(pattern.lower(), start)
                            if pos == -1:
                                break

                            entity_info = {
                                'text': pattern,
                                'start': pos,
                                'end': pos + len(pattern),
                                'category': category
                            }
                            esg_entities[category].append(entity_info)
                            start = pos + 1

            return esg_entities

        except Exception:
            return {
                'environmental_terms': [], 'social_terms': [], 'governance_terms': [],
                'sustainability_metrics': [], 'regulatory_bodies': [], 'esg_ratings_agencies': []
            }

    def _calculate_entity_statistics(self, entities: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calculate statistics about extracted entities."""
        try:
            stats = {
                'total_entities': sum(len(ents) for ents in entities.values()),
                'entity_distribution': {k: len(v) for k, v in entities.items()},
                'most_common_type': max(entities.keys(), key=lambda k: len(entities[k])) if entities else None
            }

            # Calculate entity diversity
            non_empty_types = sum(1 for ents in entities.values() if len(ents) > 0)
            stats['entity_type_diversity'] = non_empty_types / len(entities) if entities else 0

            return stats

        except Exception:
            return {}

    def _analyze_esg_entities(self, esg_entities: Dict[str, List[Any]], text: str) -> Dict[str, Any]:
        """Analyze ESG-specific entity patterns."""
        try:
            analysis = {
                'esg_coverage': {},
                'dominant_theme': None,
                'esg_completeness': 0
            }

            # Calculate coverage for each ESG category
            total_esg_entities = sum(len(entities) for entities in esg_entities.values())

            for category, entities in esg_entities.items():
                if category in ['environmental_terms', 'social_terms', 'governance_terms']:
                    coverage = len(entities) / max(total_esg_entities, 1)
                    analysis['esg_coverage'][category] = round(coverage, self.config.precision)

            # Determine dominant theme
            if analysis['esg_coverage']:
                dominant_theme = max(analysis['esg_coverage'], key=analysis['esg_coverage'].get)
                analysis['dominant_theme'] = dominant_theme

            # ESG completeness (presence of all three pillars)
            esg_pillars = ['environmental_terms', 'social_terms', 'governance_terms']
            present_pillars = sum(1 for pillar in esg_pillars if len(esg_entities.get(pillar, [])) > 0)
            analysis['esg_completeness'] = present_pillars / len(esg_pillars)

            return analysis

        except Exception:
            return {}

    def _classify_relationship(self, verb: str) -> str:
        """Classify relationship type based on verb."""
        verb_lower = verb.lower()

        if verb_lower in ['invest', 'fund', 'finance', 'support']:
            return 'financial'
        elif verb_lower in ['partner', 'collaborate', 'work', 'join']:
            return 'partnership'
        elif verb_lower in ['acquire', 'merge', 'buy', 'purchase']:
            return 'acquisition'
        elif verb_lower in ['report', 'announce', 'disclose', 'publish']:
            return 'disclosure'
        else:
            return 'other'

    def _analyze_entity_cooccurrence(self, entities: List[Tuple], text: str) -> Dict[str, Any]:
        """Analyze which entities appear together."""
        try:
            cooccurrence = {}
            sentences = sent_tokenize(text)

            for sentence in sentences:
                sentence_entities = []
                sentence_lower = sentence.lower()

                for entity_text, entity_label, start, end in entities:
                    if entity_text.lower() in sentence_lower:
                        sentence_entities.append(entity_text)

                # Record co-occurrences
                for i, ent1 in enumerate(sentence_entities):
                    for ent2 in sentence_entities[i+1:]:
                        pair = tuple(sorted([ent1, ent2]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1

            # Sort by frequency
            sorted_cooccurrence = dict(sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True))

            return {
                'cooccurrence_pairs': dict(list(sorted_cooccurrence.items())[:10]),  # Top 10
                'total_pairs': len(sorted_cooccurrence)
            }

        except Exception:
            return {}

    def _calculate_entity_sentiment_trend(self, sentiment_scores: List[float]) -> str:
        """Calculate sentiment trend for an entity."""
        try:
            if len(sentiment_scores) < 2:
                return 'insufficient_data'

            # Simple trend calculation
            recent_half = sentiment_scores[len(sentiment_scores)//2:]
            early_half = sentiment_scores[:len(sentiment_scores)//2]

            recent_avg = np.mean(recent_half)
            early_avg = np.mean(early_half)

            diff = recent_avg - early_avg

            if diff > 0.1:
                return 'improving'
            elif diff < -0.1:
                return 'declining'
            else:
                return 'stable'

        except Exception:
            return 'unknown'


# Example usage and testing functions
def example_data_scraping():
    """Example of data scraping functionality."""
    scraper = DataScraper()

    # Example company list
    companies = ['Apple', 'Microsoft', 'Tesla']

    # Scrape news articles
    news_result = scraper.scrape_news_articles(companies)
    print(f"Scraped {news_result.get('esg_relevant', 0)} ESG-relevant articles")

    # Get financial news feeds
    feed_result = scraper.get_financial_news_feeds(esg_focused=True)
    print(f"Collected {feed_result.get('esg_articles', 0)} articles from feeds")

    return news_result


def example_text_preprocessing():
    """Example of text preprocessing and TF-IDF."""
    sample_texts = [
        "Apple announced new sustainability initiatives focusing on carbon neutrality and renewable energy.",
        "Microsoft's diversity and inclusion programs have improved workplace equality and employee satisfaction.",
        "Tesla's governance structure includes independent board members overseeing executive compensation."
    ]

    preprocessor = TextPreprocessor()

    # Create document-term matrix
    dtm_result = preprocessor.create_document_term_matrix(sample_texts, method='tfidf')

    if 'error' not in dtm_result:
        print(f"Created DTM with shape: {dtm_result['matrix_shape']}")
        print(f"Top features: {list(dtm_result['feature_importance'].keys())[:5]}")

    return dtm_result


def example_sentiment_analysis():
    """Example of ESG sentiment analysis."""
    sample_text = """
    Apple has made significant progress in environmental sustainability, achieving carbon neutrality
    across its operations. The company's commitment to renewable energy and reducing its carbon
    footprint demonstrates strong environmental leadership. However, concerns remain about labor
    practices in its supply chain and the need for greater transparency in governance structures.
    """

    analyzer = SentimentAnalyzer()

    # ESG sentiment analysis
    esg_sentiment = analyzer.esg_sentiment_analysis(sample_text)

    if 'error' not in esg_sentiment:
        print("ESG Sentiment Analysis:")
        for category, sentiment in esg_sentiment['esg_category_sentiments'].items():
            print(f"{category}: {sentiment['classification']} ({sentiment['sentiment_score']:.3f})")

    return esg_sentiment


def example_topic_modeling():
    """Example of topic modeling."""
    sample_documents = [
        "Climate change initiatives and carbon reduction strategies",
        "Diversity programs and workplace inclusion efforts",
        "Corporate governance and board independence",
        "Renewable energy investments and sustainability reporting",
        "Employee welfare and human rights protection",
        "Executive compensation and shareholder rights"
    ]

    topic_modeler = TopicModeling()

    # LDA topic modeling
    lda_result = topic_modeler.latent_dirichlet_allocation(sample_documents, n_topics=3)

    if 'error' not in lda_result:
        print("Discovered Topics:")
        for topic in lda_result['topics']:
            words = [w['word'] for w in topic['words'][:5]]
            print(f"Topic {topic['topic_id']}: {', '.join(words)}")

    return lda_result


if __name__ == "__main__":
    print("ESG NLP Analysis Framework")
    print("=" * 50)

    try:
        # Example data scraping
        print("\n1. Data Scraping Example:")
        scraping_result = example_data_scraping()

        # Example text preprocessing
        print("\n2. Text Preprocessing Example:")
        preprocessing_result = example_text_preprocessing()

        # Example sentiment analysis
        print("\n3. Sentiment Analysis Example:")
        sentiment_result = example_sentiment_analysis()

        # Example topic modeling
        print("\n4. Topic Modeling Example:")
        topic_result = example_topic_modeling()

        print("\nESG NLP analysis framework ready for use!")

    except Exception as e:
        print(f"Error in examples: {e}")


if __name__ == "__main__":
    main()