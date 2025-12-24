"""
Data Access Module for Economic and Financial Data

This module provides functions for retrieving data from various economic and financial data sources:
- Econdb
- FRED (Federal Reserve Economic Data)
- OECDStat
- Eurostat
- WorldBank WDI (World Development Indicators)
- Yahoo Finance
- Google Finance
- Investing.com
- Fama/French Library
- FINRA Markets (mutual fund data)
- Morningstar
- Thrift Savings Plan (TSP)
- IEX (Investors Exchange)
- Moscow Exchange (MOEX)
- Nasdaq
- Naver Finance (Korean exchanges)

Dependencies will be imported as needed for each data source.
"""

import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import warnings

# FRED (Federal Reserve Economic Data) Functions
def get_fred_data(series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                  api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from FRED (Federal Reserve Economic Data)

    Args:
        series_id: FRED series identifier
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: FRED API key

    Returns:
        DataFrame with date index and series data
    """
    try:
        import fredapi
        fred = fredapi.Fred(api_key=api_key)
        data = fred.get_series(series_id, start=start_date, end=end_date)
        return data.to_frame(name=series_id)
    except ImportError:
        raise ImportError("fredapi package required. Install with: pip install fredapi")
    except Exception as e:
        raise Exception(f"Error retrieving FRED data: {str(e)}")


# Yahoo Finance Functions
def get_yahoo_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Retrieve stock data from Yahoo Finance

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except ImportError:
        raise ImportError("yfinance package required. Install with: pip install yfinance")
    except Exception as e:
        raise Exception(f"Error retrieving Yahoo Finance data: {str(e)}")


def get_yahoo_info(symbol: str) -> Dict[str, Any]:
    """
    Get company information from Yahoo Finance

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with company information
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return ticker.info
    except ImportError:
        raise ImportError("yfinance package required. Install with: pip install yfinance")
    except Exception as e:
        raise Exception(f"Error retrieving Yahoo Finance info: {str(e)}")


# WorldBank WDI Functions
def get_worldbank_data(indicator: str, country: str = "all", start_year: Optional[int] = None,
                       end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieve data from WorldBank World Development Indicators

    Args:
        indicator: WDI indicator code
        country: Country code or 'all' for all countries
        start_year: Start year
        end_year: End year

    Returns:
        DataFrame with WorldBank data
    """
    try:
        import wbdata
        data = wbdata.get_dataframe({indicator: indicator},
                                  country=country,
                                  data_date=(start_year, end_year) if start_year and end_year else None)
        return data
    except ImportError:
        raise ImportError("wbdata package required. Install with: pip install wbdata")
    except Exception as e:
        raise Exception(f"Error retrieving WorldBank data: {str(e)}")


# OECD Functions
def get_oecd_data(dataset: str, dimensions: Optional[Dict[str, str]] = None,
                  start_period: Optional[str] = None, end_period: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from OECD Statistics

    Args:
        dataset: OECD dataset identifier
        dimensions: Dictionary of dimension filters
        start_period: Start period
        end_period: End period

    Returns:
        DataFrame with OECD data
    """
    try:
        import pandasdmx as sdmx
        oecd = sdmx.Request('OECD')

        params = {}
        if dimensions:
            params.update(dimensions)
        if start_period:
            params['startPeriod'] = start_period
        if end_period:
            params['endPeriod'] = end_period

        data_msg = oecd.data(dataset, params=params)
        data = data_msg.data
        return sdmx.to_pandas(data)
    except ImportError:
        raise ImportError("pandasdmx package required. Install with: pip install pandasdmx")
    except Exception as e:
        raise Exception(f"Error retrieving OECD data: {str(e)}")


# Eurostat Functions
def get_eurostat_data(dataset: str, filters: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Retrieve data from Eurostat

    Args:
        dataset: Eurostat dataset code
        filters: Dictionary of filters to apply

    Returns:
        DataFrame with Eurostat data
    """
    try:
        import eurostat
        data = eurostat.get_data_df(dataset, filters=filters)
        return data
    except ImportError:
        raise ImportError("eurostat package required. Install with: pip install eurostat")
    except Exception as e:
        raise Exception(f"Error retrieving Eurostat data: {str(e)}")


# Econdb Functions
def get_econdb_data(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                    api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from Econdb

    Args:
        ticker: Econdb ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Econdb API key

    Returns:
        DataFrame with Econdb data
    """
    base_url = "https://www.econdb.com/api/series"

    params = {
        "ticker": ticker,
        "format": "json"
    }

    if start_date:
        params["from"] = start_date
    if end_date:
        params["to"] = end_date
    if api_key:
        params["token"] = api_key

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        if "data" in data:
            df = pd.DataFrame(data["data"])
            if "date" in df.columns and "value" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return df

        raise ValueError("Unexpected data format from Econdb API")

    except requests.RequestException as e:
        raise Exception(f"Error retrieving Econdb data: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing Econdb data: {str(e)}")


# Google Finance Functions (Note: Google Finance API is deprecated)
def get_google_finance_data(symbol: str) -> pd.DataFrame:
    """
    Retrieve data from Google Finance (Limited functionality - API deprecated)

    Args:
        symbol: Stock symbol

    Returns:
        DataFrame with basic stock data
    """
    warnings.warn("Google Finance API is deprecated. Consider using Yahoo Finance or other alternatives.",
                  DeprecationWarning)

    # Basic implementation using web scraping (may be unreliable)
    try:
        url = f"https://www.google.com/finance/quote/{symbol}"
        # This would require additional web scraping implementation
        # For now, raising a NotImplementedError
        raise NotImplementedError("Google Finance API is no longer available. Use Yahoo Finance instead.")
    except Exception as e:
        raise Exception(f"Error retrieving Google Finance data: {str(e)}")


# Investing.com Functions
def get_investing_data(symbol: str, country: str = "united states",
                       from_date: Optional[str] = None, to_date: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from Investing.com

    Args:
        symbol: Symbol name on Investing.com
        country: Country name
        from_date: Start date in DD/MM/YYYY format
        to_date: End date in DD/MM/YYYY format

    Returns:
        DataFrame with historical data
    """
    try:
        import investpy

        if from_date and to_date:
            data = investpy.get_stock_historical_data(stock=symbol,
                                                    country=country,
                                                    from_date=from_date,
                                                    to_date=to_date)
        else:
            data = investpy.get_stock_recent_data(stock=symbol, country=country)

        return data
    except ImportError:
        raise ImportError("investpy package required. Install with: pip install investpy")
    except Exception as e:
        raise Exception(f"Error retrieving Investing.com data: {str(e)}")


# Fama/French Library Functions
def get_famafrench_data(dataset_name: str) -> pd.DataFrame:
    """
    Retrieve data from Fama/French library

    Args:
        dataset_name: Name of the Fama/French dataset (e.g., 'F-F_Research_Data_Factors')

    Returns:
        DataFrame with Fama/French factor data
    """
    try:
        import pandas_datareader.famafrench as web
        data = web.FamaFrenchReader(dataset_name).read()

        # If multiple datasets in the result, return the first one
        if isinstance(data, dict):
            return list(data.values())[0]
        return data
    except ImportError:
        raise ImportError("pandas-datareader package required. Install with: pip install pandas-datareader")
    except Exception as e:
        raise Exception(f"Error retrieving Fama/French data: {str(e)}")


def list_famafrench_datasets() -> List[str]:
    """
    List available Fama/French datasets

    Returns:
        List of available dataset names
    """
    try:
        import pandas_datareader.famafrench as web
        return list(web.get_available_datasets().keys())
    except ImportError:
        raise ImportError("pandas-datareader package required. Install with: pip install pandas-datareader")
    except Exception as e:
        raise Exception(f"Error listing Fama/French datasets: {str(e)}")


# FINRA Markets Functions
def get_finra_mutual_fund_data(fund_symbol: str) -> pd.DataFrame:
    """
    Retrieve mutual fund data from FINRA Markets

    Args:
        fund_symbol: Mutual fund symbol

    Returns:
        DataFrame with mutual fund data
    """
    base_url = "https://tools.finra.org/fund_analyzer/1/FundService.asmx/GetFund"

    params = {
        "symbol": fund_symbol
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        # Parse XML response (FINRA returns XML)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        # Extract fund data (basic implementation)
        fund_data = {}
        for element in root.iter():
            if element.text and element.tag:
                fund_data[element.tag] = element.text

        return pd.DataFrame([fund_data])
    except ImportError:
        raise ImportError("xml.etree.ElementTree required (part of standard library)")
    except Exception as e:
        raise Exception(f"Error retrieving FINRA mutual fund data: {str(e)}")


# Morningstar Functions
def get_morningstar_data(ticker: str, data_type: str = "price") -> pd.DataFrame:
    """
    Retrieve data from Morningstar

    Args:
        ticker: Stock or fund ticker
        data_type: Type of data ('price', 'fundamentals', 'portfolio')

    Returns:
        DataFrame with Morningstar data
    """
    try:
        import morningstar_data as ms

        if data_type == "price":
            data = ms.get_price_data(ticker)
        elif data_type == "fundamentals":
            data = ms.get_fundamental_data(ticker)
        elif data_type == "portfolio":
            data = ms.get_portfolio_data(ticker)
        else:
            raise ValueError("data_type must be 'price', 'fundamentals', or 'portfolio'")

        return data
    except ImportError:
        # Fallback to web scraping approach
        base_url = f"https://www.morningstar.com/stocks/xnas/{ticker}/quote"
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            # Basic web scraping implementation would go here
            raise NotImplementedError("Morningstar web scraping not fully implemented. Consider using morningstar_data package.")
        except Exception as e:
            raise Exception(f"Error retrieving Morningstar data: {str(e)}")


# Thrift Savings Plan (TSP) Functions
def get_tsp_data(fund: str = "all", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve Thrift Savings Plan (TSP) fund data

    Args:
        fund: TSP fund ('G', 'F', 'C', 'S', 'I', 'L2025', 'L2030', etc.) or 'all'
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with TSP fund performance data
    """
    base_url = "https://www.tsp.gov/data/fund-performance"

    try:
        # TSP provides CSV data
        response = requests.get(f"{base_url}/monthly-returns.csv")
        response.raise_for_status()

        from io import StringIO
        data = pd.read_csv(StringIO(response.text))

        # Filter by fund if specified
        if fund != "all" and fund in data.columns:
            data = data[['Date', fund]]

        # Convert date column
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

        # Filter by date range
        if start_date and end_date:
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data.loc[mask]

        return data
    except Exception as e:
        raise Exception(f"Error retrieving TSP data: {str(e)}")


# IEX (Investors Exchange) Functions
def get_iex_data(symbol: str, data_type: str = "quote", api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from IEX (Investors Exchange)

    Args:
        symbol: Stock symbol
        data_type: Type of data ('quote', 'chart', 'stats', 'company')
        api_key: IEX API key

    Returns:
        DataFrame with IEX data
    """
    try:
        import pyEX

        client = pyEX.Client(api_key)

        if data_type == "quote":
            data = client.quote(symbol)
        elif data_type == "chart":
            data = client.chart(symbol)
        elif data_type == "stats":
            data = client.stats(symbol)
        elif data_type == "company":
            data = client.company(symbol)
        else:
            raise ValueError("data_type must be 'quote', 'chart', 'stats', or 'company'")

        return pd.DataFrame(data)
    except ImportError:
        # Fallback to direct API calls
        base_url = "https://cloud.iexapis.com/stable"

        params = {}
        if api_key:
            params['token'] = api_key

        try:
            response = requests.get(f"{base_url}/stock/{symbol}/{data_type}", params=params)
            response.raise_for_status()

            data = response.json()
            return pd.DataFrame([data] if isinstance(data, dict) else data)
        except Exception as e:
            raise Exception(f"Error retrieving IEX data: {str(e)}")


# Moscow Exchange (MOEX) Functions
def get_moex_data(symbol: str, board: str = "TQBR", start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from Moscow Exchange (MOEX)

    Args:
        symbol: Stock symbol (e.g., 'SBER')
        board: Trading board ('TQBR' for main board)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with MOEX historical data
    """
    try:
        import moexalgo

        data = moexalgo.Ticker(symbol)
        candles = data.candles(start=start_date, end=end_date, period="1D")
        return candles
    except ImportError:
        # Fallback to direct API calls
        base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/{}/securities/{}/candles.json"

        params = {
            "interval": "1",
            "from": start_date,
            "till": end_date
        }

        try:
            response = requests.get(base_url.format(board, symbol), params=params)
            response.raise_for_status()

            data = response.json()
            candles_data = data['candles']['data']
            columns = data['candles']['columns']

            df = pd.DataFrame(candles_data, columns=columns)
            if 'begin' in df.columns:
                df['begin'] = pd.to_datetime(df['begin'])
                df.set_index('begin', inplace=True)

            return df
        except Exception as e:
            raise Exception(f"Error retrieving MOEX data: {str(e)}")


# Nasdaq Functions
def get_nasdaq_data(symbol: str, data_type: str = "historical") -> pd.DataFrame:
    """
    Retrieve data from Nasdaq

    Args:
        symbol: Stock symbol
        data_type: Type of data ('historical', 'realtime', 'company')

    Returns:
        DataFrame with Nasdaq data
    """
    try:
        import nasdaq_stock_screener as nss

        if data_type == "historical":
            data = nss.get_historical_data(symbol)
        elif data_type == "company":
            data = nss.get_company_info(symbol)
        else:
            raise ValueError("data_type must be 'historical' or 'company'")

        return data
    except ImportError:
        # Fallback to Nasdaq API
        base_url = "https://api.nasdaq.com/api/quote/{}/historical"

        params = {
            "assetclass": "stocks",
            "fromdate": "2023-01-01",
            "limit": 9999,
            "todate": datetime.now().strftime("%Y-%m-%d")
        }

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(base_url.format(symbol), params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            if 'data' in data and 'tradesTable' in data['data']:
                trades = data['data']['tradesTable']['rows']
                df = pd.DataFrame(trades)

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

                return df
            else:
                raise ValueError("Unexpected response format from Nasdaq API")
        except Exception as e:
            raise Exception(f"Error retrieving Nasdaq data: {str(e)}")


# Naver Finance (Korean Exchanges) Functions
def get_naver_finance_data(symbol: str, market: str = "KOSPI") -> pd.DataFrame:
    """
    Retrieve data from Naver Finance (Korean exchanges)

    Args:
        symbol: Stock symbol (e.g., '005930' for Samsung)
        market: Market type ('KOSPI', 'KOSDAQ')

    Returns:
        DataFrame with Korean stock data
    """
    try:
        import pykrx

        if market.upper() == "KOSPI":
            data = pykrx.stock.get_market_ohlcv_by_date("20200101", "20231231", symbol)
        elif market.upper() == "KOSDAQ":
            data = pykrx.stock.get_market_ohlcv_by_date("20200101", "20231231", symbol, "KOSDAQ")
        else:
            raise ValueError("market must be 'KOSPI' or 'KOSDAQ'")

        return data
    except ImportError:
        # Fallback to Naver Finance web scraping
        base_url = f"https://finance.naver.com/item/sise_day.nhn?code={symbol}"

        try:
            response = requests.get(base_url)
            response.raise_for_status()

            # Parse HTML response using pandas
            tables = pd.read_html(response.text)

            if tables:
                data = tables[0]
                # Clean and process the data
                data = data.dropna()
                if '날짜' in data.columns:
                    data['날짜'] = pd.to_datetime(data['날짜'])
                    data.set_index('날짜', inplace=True)

                return data
            else:
                raise ValueError("No data tables found")
        except Exception as e:
            raise Exception(f"Error retrieving Naver Finance data: {str(e)}")


# Tiingo Functions
def get_tiingo_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                    api_key: Optional[str] = None, data_type: str = "daily") -> pd.DataFrame:
    """
    Retrieve data from Tiingo

    Args:
        symbol: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Tiingo API key
        data_type: Type of data ('daily', 'intraday', 'fundamentals')

    Returns:
        DataFrame with Tiingo data
    """
    if not api_key:
        raise ValueError("Tiingo API key is required")

    base_url = "https://api.tiingo.com/tiingo"

    if data_type == "daily":
        endpoint = f"{base_url}/daily/{symbol}/prices"
    elif data_type == "intraday":
        endpoint = f"{base_url}/iex/{symbol}/prices"
    elif data_type == "fundamentals":
        endpoint = f"{base_url}/fundamentals/{symbol}/daily"
    else:
        raise ValueError("data_type must be 'daily', 'intraday', or 'fundamentals'")

    params = {
        "token": api_key,
        "format": "json"
    }

    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df
    except Exception as e:
        raise Exception(f"Error retrieving Tiingo data: {str(e)}")


def get_tiingo_crypto_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                          api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve cryptocurrency data from Tiingo

    Args:
        symbol: Crypto symbol (e.g., 'btcusd')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Tiingo API key

    Returns:
        DataFrame with cryptocurrency data
    """
    if not api_key:
        raise ValueError("Tiingo API key is required")

    base_url = f"https://api.tiingo.com/tiingo/crypto/prices"

    params = {
        "tickers": symbol,
        "token": api_key,
        "format": "json"
    }

    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()
        if data and len(data) > 0:
            df = pd.DataFrame(data[0]['priceData'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        else:
            raise ValueError("No data returned from Tiingo")
    except Exception as e:
        raise Exception(f"Error retrieving Tiingo crypto data: {str(e)}")


# Quandl Functions
def get_quandl_data(database_code: str, dataset_code: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from Quandl (now part of Nasdaq Data Link)

    Args:
        database_code: Quandl database code (e.g., 'WIKI')
        dataset_code: Dataset code (e.g., 'AAPL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Quandl/Nasdaq Data Link API key

    Returns:
        DataFrame with Quandl data
    """
    try:
        import quandl

        if api_key:
            quandl.ApiConfig.api_key = api_key

        data = quandl.get(f"{database_code}/{dataset_code}",
                         start_date=start_date,
                         end_date=end_date)
        return data
    except ImportError:
        # Fallback to direct API calls
        base_url = f"https://data.nasdaq.com/api/v3/datasets/{database_code}/{dataset_code}/data.json"

        params = {}
        if api_key:
            params["api_key"] = api_key
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            dataset_data = data['dataset_data']

            df = pd.DataFrame(dataset_data['data'], columns=dataset_data['column_names'])
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            return df
        except Exception as e:
            raise Exception(f"Error retrieving Quandl data: {str(e)}")


# Alpha Vantage Functions
def get_alphavantage_data(symbol: str, function: str = "TIME_SERIES_DAILY",
                         api_key: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Retrieve data from Alpha Vantage

    Args:
        symbol: Stock symbol
        function: Alpha Vantage function ('TIME_SERIES_DAILY', 'TIME_SERIES_INTRADAY', etc.)
        api_key: Alpha Vantage API key
        **kwargs: Additional parameters for the API call

    Returns:
        DataFrame with Alpha Vantage data
    """
    if not api_key:
        raise ValueError("Alpha Vantage API key is required")

    base_url = "https://www.alphavantage.co/query"

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "datatype": "json"
    }
    params.update(kwargs)

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        # Handle different response formats
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key or "Daily" in key or "Intraday" in key:
                time_series_key = key
                break

        if time_series_key and time_series_key in data:
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df
        else:
            raise ValueError(f"Unexpected response format from Alpha Vantage: {list(data.keys())}")

    except Exception as e:
        raise Exception(f"Error retrieving Alpha Vantage data: {str(e)}")


def get_alphavantage_forex(from_currency: str, to_currency: str, api_key: Optional[str] = None,
                          function: str = "FX_DAILY") -> pd.DataFrame:
    """
    Retrieve forex data from Alpha Vantage

    Args:
        from_currency: Base currency (e.g., 'USD')
        to_currency: Quote currency (e.g., 'EUR')
        api_key: Alpha Vantage API key
        function: Function type ('FX_DAILY', 'FX_INTRADAY')

    Returns:
        DataFrame with forex data
    """
    return get_alphavantage_data(f"{from_currency}{to_currency}", function=function, api_key=api_key)


# Enigma Functions
def get_enigma_data(dataset_id: str, api_key: Optional[str] = None, **filters) -> pd.DataFrame:
    """
    Retrieve data from Enigma

    Args:
        dataset_id: Enigma dataset identifier
        api_key: Enigma API key
        **filters: Additional filters for the dataset

    Returns:
        DataFrame with Enigma data
    """
    if not api_key:
        raise ValueError("Enigma API key is required")

    base_url = f"https://api.enigma.com/v2/data/{dataset_id}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    params = filters

    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if 'result' in data:
            df = pd.DataFrame(data['result'])
            return df
        else:
            raise ValueError("Unexpected response format from Enigma")

    except Exception as e:
        raise Exception(f"Error retrieving Enigma data: {str(e)}")


# Stooq Functions
def get_stooq_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve data from Stooq

    Args:
        symbol: Stock symbol (e.g., 'AAPL.US')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with Stooq historical data
    """
    try:
        import pandas_datareader.stooq as web

        data = web.StooqDailyReader(symbol, start=start_date, end=end_date).read()
        return data
    except ImportError:
        # Fallback to direct CSV download
        base_url = "https://stooq.com/q/d/l/"

        params = {
            "s": symbol,
            "i": "d"  # daily data
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter by date range if specified
            if start_date and end_date:
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df.loc[mask]

            return df
        except Exception as e:
            raise Exception(f"Error retrieving Stooq data: {str(e)}")


# Yahoo Finance Options Functions
def get_yahoo_options_chain(symbol: str, expiration_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Retrieve options chain data from Yahoo Finance

    Args:
        symbol: Stock symbol
        expiration_date: Expiration date in YYYY-MM-DD format (if None, gets nearest expiration)

    Returns:
        Dictionary with 'calls' and 'puts' DataFrames
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)

        if expiration_date:
            options_chain = ticker.option_chain(expiration_date)
        else:
            # Get the nearest expiration date
            exp_dates = ticker.options
            if exp_dates:
                options_chain = ticker.option_chain(exp_dates[0])
            else:
                raise ValueError("No options expiration dates available")

        return {
            'calls': options_chain.calls,
            'puts': options_chain.puts
        }
    except ImportError:
        raise ImportError("yfinance package required. Install with: pip install yfinance")
    except Exception as e:
        raise Exception(f"Error retrieving Yahoo Finance options data: {str(e)}")


def get_yahoo_options_summary(symbol: str, expiration_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get options summary metrics from Yahoo Finance

    Args:
        symbol: Stock symbol
        expiration_date: Expiration date in YYYY-MM-DD format

    Returns:
        Dictionary with options summary metrics
    """
    try:
        options_data = get_yahoo_options_chain(symbol, expiration_date)
        calls_df = options_data['calls']
        puts_df = options_data['puts']

        # Calculate summary metrics
        summary = {
            'symbol': symbol,
            'expiration_date': expiration_date,
            'total_call_volume': calls_df['volume'].sum() if 'volume' in calls_df.columns else 0,
            'total_put_volume': puts_df['volume'].sum() if 'volume' in puts_df.columns else 0,
            'total_call_open_interest': calls_df['openInterest'].sum() if 'openInterest' in calls_df.columns else 0,
            'total_put_open_interest': puts_df['openInterest'].sum() if 'openInterest' in puts_df.columns else 0,
        }

        # Calculate put/call ratios
        if summary['total_call_volume'] > 0:
            summary['put_call_volume_ratio'] = summary['total_put_volume'] / summary['total_call_volume']
        else:
            summary['put_call_volume_ratio'] = None

        if summary['total_call_open_interest'] > 0:
            summary['put_call_oi_ratio'] = summary['total_put_open_interest'] / summary['total_call_open_interest']
        else:
            summary['put_call_oi_ratio'] = None

        return summary

    except Exception as e:
        raise Exception(f"Error calculating options summary: {str(e)}")


def get_yahoo_options_by_strike(symbol: str, strike_min: float, strike_max: float,
                               expiration_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Get options data filtered by strike price range

    Args:
        symbol: Stock symbol
        strike_min: Minimum strike price
        strike_max: Maximum strike price
        expiration_date: Expiration date in YYYY-MM-DD format

    Returns:
        Dictionary with filtered 'calls' and 'puts' DataFrames
    """
    try:
        options_data = get_yahoo_options_chain(symbol, expiration_date)

        # Filter by strike price range
        calls_filtered = options_data['calls']
        puts_filtered = options_data['puts']

        if 'strike' in calls_filtered.columns:
            calls_filtered = calls_filtered[
                (calls_filtered['strike'] >= strike_min) &
                (calls_filtered['strike'] <= strike_max)
            ]

        if 'strike' in puts_filtered.columns:
            puts_filtered = puts_filtered[
                (puts_filtered['strike'] >= strike_min) &
                (puts_filtered['strike'] <= strike_max)
            ]

        return {
            'calls': calls_filtered,
            'puts': puts_filtered
        }

    except Exception as e:
        raise Exception(f"Error filtering options by strike: {str(e)}")


def get_yahoo_options_expiration_dates(symbol: str) -> List[str]:
    """
    Get all available options expiration dates for a symbol

    Args:
        symbol: Stock symbol

    Returns:
        List of expiration dates
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        return list(ticker.options)
    except ImportError:
        raise ImportError("yfinance package required. Install with: pip install yfinance")
    except Exception as e:
        raise Exception(f"Error retrieving options expiration dates: {str(e)}")


# File Input Functions
def read_csv_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Read data from CSV file

    Args:
        file_path: Path to CSV file
        **kwargs: Additional parameters for pd.read_csv()

    Returns:
        DataFrame with CSV data
    """
    try:
        # Set default parameters
        default_params = {
            'encoding': 'utf-8',
            'low_memory': False
        }
        default_params.update(kwargs)

        df = pd.read_csv(file_path, **default_params)

        # Try to parse date columns automatically
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        return df
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")


def read_excel_file(file_path: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Read data from Excel file (XLS or XLSX)

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name to read (if None, reads first sheet)
        **kwargs: Additional parameters for pd.read_excel()

    Returns:
        DataFrame with Excel data
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

        # Try to parse date columns automatically
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        return df
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")


def read_json_file(file_path: str, orient: str = 'records', **kwargs) -> pd.DataFrame:
    """
    Read data from JSON file

    Args:
        file_path: Path to JSON file
        orient: JSON orientation ('records', 'index', 'values', 'split', 'table')
        **kwargs: Additional parameters for pd.read_json()

    Returns:
        DataFrame with JSON data
    """
    try:
        df = pd.read_json(file_path, orient=orient, **kwargs)

        # Try to parse date columns automatically
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        return df
    except Exception as e:
        raise Exception(f"Error reading JSON file: {str(e)}")


def list_excel_sheets(file_path: str) -> List[str]:
    """
    List all sheet names in an Excel file

    Args:
        file_path: Path to Excel file

    Returns:
        List of sheet names
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
    except Exception as e:
        raise Exception(f"Error listing Excel sheets: {str(e)}")


# SQLite Database Functions
def read_sqlite_data(db_path: str, query: str, **kwargs) -> pd.DataFrame:
    """
    Read data from SQLite database

    Args:
        db_path: Path to SQLite database file
        query: SQL query to execute
        **kwargs: Additional parameters for pd.read_sql()

    Returns:
        DataFrame with query results
    """
    try:
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(query, conn, **kwargs)
            return df
    except ImportError:
        raise ImportError("sqlite3 package required (part of standard library)")
    except Exception as e:
        raise Exception(f"Error reading from SQLite database: {str(e)}")


def read_sqlite_table(db_path: str, table_name: str, **kwargs) -> pd.DataFrame:
    """
    Read entire table from SQLite database

    Args:
        db_path: Path to SQLite database file
        table_name: Name of table to read
        **kwargs: Additional parameters for pd.read_sql()

    Returns:
        DataFrame with table data
    """
    query = f"SELECT * FROM {table_name}"
    return read_sqlite_data(db_path, query, **kwargs)


def list_sqlite_tables(db_path: str) -> List[str]:
    """
    List all tables in SQLite database

    Args:
        db_path: Path to SQLite database file

    Returns:
        List of table names
    """
    try:
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            return tables
    except Exception as e:
        raise Exception(f"Error listing SQLite tables: {str(e)}")


# MongoDB Functions
def read_mongodb_data(connection_string: str, database: str, collection: str,
                     query: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
    """
    Read data from MongoDB

    Args:
        connection_string: MongoDB connection string
        database: Database name
        collection: Collection name
        query: MongoDB query filter (if None, returns all documents)
        **kwargs: Additional parameters

    Returns:
        DataFrame with MongoDB data
    """
    try:
        import pymongo

        client = pymongo.MongoClient(connection_string)
        db = client[database]
        coll = db[collection]

        if query is None:
            query = {}

        # Execute query and convert to DataFrame
        cursor = coll.find(query)
        data = list(cursor)

        if data:
            df = pd.DataFrame(data)

            # Convert ObjectId to string if present
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)

            return df
        else:
            return pd.DataFrame()

    except ImportError:
        raise ImportError("pymongo package required. Install with: pip install pymongo")
    except Exception as e:
        raise Exception(f"Error reading from MongoDB: {str(e)}")


def list_mongodb_collections(connection_string: str, database: str) -> List[str]:
    """
    List all collections in MongoDB database

    Args:
        connection_string: MongoDB connection string
        database: Database name

    Returns:
        List of collection names
    """
    try:
        import pymongo

        client = pymongo.MongoClient(connection_string)
        db = client[database]
        return db.list_collection_names()
    except ImportError:
        raise ImportError("pymongo package required. Install with: pip install pymongo")
    except Exception as e:
        raise Exception(f"Error listing MongoDB collections: {str(e)}")


# ETL Class
class DataETL:
    """
    ETL (Extract, Transform, Load) class for data processing and database operations
    """

    def __init__(self):
        self.data = None
        self.transformations = []

    # Extract Methods
    def extract_from_csv(self, file_path: str, **kwargs) -> 'DataETL':
        """Extract data from CSV file"""
        self.data = read_csv_file(file_path, **kwargs)
        return self

    def extract_from_excel(self, file_path: str, sheet_name: Optional[str] = None, **kwargs) -> 'DataETL':
        """Extract data from Excel file"""
        self.data = read_excel_file(file_path, sheet_name, **kwargs)
        return self

    def extract_from_json(self, file_path: str, orient: str = 'records', **kwargs) -> 'DataETL':
        """Extract data from JSON file"""
        self.data = read_json_file(file_path, orient, **kwargs)
        return self

    def extract_from_sqlite(self, db_path: str, query: str, **kwargs) -> 'DataETL':
        """Extract data from SQLite database"""
        self.data = read_sqlite_data(db_path, query, **kwargs)
        return self

    def extract_from_mongodb(self, connection_string: str, database: str, collection: str,
                           query: Optional[Dict[str, Any]] = None, **kwargs) -> 'DataETL':
        """Extract data from MongoDB"""
        self.data = read_mongodb_data(connection_string, database, collection, query, **kwargs)
        return self

    def extract_from_api(self, api_function, *args, **kwargs) -> 'DataETL':
        """Extract data using any API function from this module"""
        self.data = api_function(*args, **kwargs)
        return self

    # Transform Methods
    def filter_data(self, condition) -> 'DataETL':
        """Filter data based on condition"""
        if self.data is not None:
            self.data = self.data[condition]
            self.transformations.append(f"Filtered data with condition")
        return self

    def select_columns(self, columns: List[str]) -> 'DataETL':
        """Select specific columns"""
        if self.data is not None:
            self.data = self.data[columns]
            self.transformations.append(f"Selected columns: {columns}")
        return self

    def rename_columns(self, column_mapping: Dict[str, str]) -> 'DataETL':
        """Rename columns"""
        if self.data is not None:
            self.data = self.data.rename(columns=column_mapping)
            self.transformations.append(f"Renamed columns: {column_mapping}")
        return self

    def add_calculated_column(self, column_name: str, calculation) -> 'DataETL':
        """Add a calculated column"""
        if self.data is not None:
            self.data[column_name] = calculation(self.data)
            self.transformations.append(f"Added calculated column: {column_name}")
        return self

    def fill_missing_values(self, method: str = 'forward', **kwargs) -> 'DataETL':
        """Fill missing values"""
        if self.data is not None:
            if method == 'forward':
                self.data = self.data.fillna(method='ffill', **kwargs)
            elif method == 'backward':
                self.data = self.data.fillna(method='bfill', **kwargs)
            elif method == 'value':
                self.data = self.data.fillna(kwargs.get('value', 0))
            elif method == 'mean':
                self.data = self.data.fillna(self.data.mean())
            elif method == 'median':
                self.data = self.data.fillna(self.data.median())

            self.transformations.append(f"Filled missing values using method: {method}")
        return self

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataETL':
        """Remove duplicate rows"""
        if self.data is not None:
            self.data = self.data.drop_duplicates(subset=subset)
            self.transformations.append("Removed duplicate rows")
        return self

    def apply_function(self, func, *args, **kwargs) -> 'DataETL':
        """Apply custom function to data"""
        if self.data is not None:
            self.data = func(self.data, *args, **kwargs)
            self.transformations.append(f"Applied custom function: {func.__name__}")
        return self

    # Load Methods
    def load_to_sqlite(self, db_path: str, table_name: str, if_exists: str = 'replace', **kwargs) -> 'DataETL':
        """Load data to SQLite database"""
        try:
            import sqlite3

            if self.data is not None:
                with sqlite3.connect(db_path) as conn:
                    self.data.to_sql(table_name, conn, if_exists=if_exists, index=False, **kwargs)
                    self.transformations.append(f"Loaded to SQLite table: {table_name}")
        except Exception as e:
            raise Exception(f"Error loading to SQLite: {str(e)}")
        return self

    def load_to_mongodb(self, connection_string: str, database: str, collection: str) -> 'DataETL':
        """Load data to MongoDB"""
        try:
            import pymongo

            if self.data is not None:
                client = pymongo.MongoClient(connection_string)
                db = client[database]
                coll = db[collection]

                # Convert DataFrame to records and insert
                records = self.data.to_dict('records')
                if records:
                    coll.insert_many(records)
                    self.transformations.append(f"Loaded to MongoDB collection: {collection}")
        except ImportError:
            raise ImportError("pymongo package required. Install with: pip install pymongo")
        except Exception as e:
            raise Exception(f"Error loading to MongoDB: {str(e)}")
        return self

    def load_to_csv(self, file_path: str, **kwargs) -> 'DataETL':
        """Load data to CSV file"""
        try:
            if self.data is not None:
                self.data.to_csv(file_path, index=False, **kwargs)
                self.transformations.append(f"Loaded to CSV file: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading to CSV: {str(e)}")
        return self

    def load_to_excel(self, file_path: str, sheet_name: str = 'Sheet1', **kwargs) -> 'DataETL':
        """Load data to Excel file"""
        try:
            if self.data is not None:
                self.data.to_excel(file_path, sheet_name=sheet_name, index=False, **kwargs)
                self.transformations.append(f"Loaded to Excel file: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading to Excel: {str(e)}")
        return self

    def load_to_json(self, file_path: str, orient: str = 'records', **kwargs) -> 'DataETL':
        """Load data to JSON file"""
        try:
            if self.data is not None:
                self.data.to_json(file_path, orient=orient, **kwargs)
                self.transformations.append(f"Loaded to JSON file: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading to JSON: {str(e)}")
        return self

    # SQL Database Support (PostgreSQL, MySQL, etc.)
    def extract_from_sql(self, connection_string: str, query: str, **kwargs) -> 'DataETL':
        """Extract data from SQL database using connection string"""
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(connection_string)
            self.data = pd.read_sql(query, engine, **kwargs)
            return self
        except ImportError:
            raise ImportError("sqlalchemy package required. Install with: pip install sqlalchemy")
        except Exception as e:
            raise Exception(f"Error extracting from SQL database: {str(e)}")

    def load_to_sql(self, connection_string: str, table_name: str, if_exists: str = 'replace', **kwargs) -> 'DataETL':
        """Load data to SQL database using connection string"""
        try:
            import sqlalchemy

            if self.data is not None:
                engine = sqlalchemy.create_engine(connection_string)
                self.data.to_sql(table_name, engine, if_exists=if_exists, index=False, **kwargs)
                self.transformations.append(f"Loaded to SQL table: {table_name}")
        except ImportError:
            raise ImportError("sqlalchemy package required. Install with: pip install sqlalchemy")
        except Exception as e:
            raise Exception(f"Error loading to SQL database: {str(e)}")
        return self

    # Utility Methods
    def get_data(self) -> pd.DataFrame:
        """Get the current data"""
        return self.data

    def get_info(self) -> Dict[str, Any]:
        """Get information about the data and transformations"""
        info = {
            'data_shape': self.data.shape if self.data is not None else None,
            'data_columns': list(self.data.columns) if self.data is not None else None,
            'transformations': self.transformations,
            'data_types': dict(self.data.dtypes) if self.data is not None else None
        }
        return info

    def preview(self, n: int = 5) -> pd.DataFrame:
        """Preview first n rows of data"""
        if self.data is not None:
            return self.data.head(n)
        return pd.DataFrame()

    def reset(self) -> 'DataETL':
        """Reset ETL pipeline"""
        self.data = None
        self.transformations = []
        return self


# Utility Functions
def list_available_sources() -> List[str]:
    """
    List all available data sources

    Returns:
        List of available data source names
    """
    return [
        "FRED",
        "Yahoo Finance",
        "WorldBank WDI",
        "OECD",
        "Eurostat",
        "Econdb",
        "Google Finance (deprecated)",
        "Investing.com",
        "Fama/French Library",
        "FINRA Markets",
        "Morningstar",
        "Thrift Savings Plan",
        "IEX",
        "Moscow Exchange",
        "Nasdaq",
        "Naver Finance",
        "Tiingo",
        "Quandl",
        "Alpha Vantage",
        "Enigma",
        "Stooq",
        "Yahoo Options",
        "CSV Files",
        "Excel Files",
        "JSON Files",
        "SQLite Database",
        "MongoDB",
        "SQL Databases"
    ]


def get_data_info() -> Dict[str, str]:
    """
    Get information about each data source

    Returns:
        Dictionary with data source descriptions
    """
    return {
        "FRED": "Federal Reserve Economic Data - US economic indicators",
        "Yahoo Finance": "Stock prices, financial data, and company information",
        "WorldBank WDI": "World Development Indicators - global development data",
        "OECD": "Organization for Economic Cooperation and Development statistics",
        "Eurostat": "European Union statistical office data",
        "Econdb": "Economic database with global economic indicators",
        "Google Finance": "Deprecated - use Yahoo Finance instead",
        "Investing.com": "Financial markets data and analysis",
        "Fama/French Library": "Academic financial research factors and portfolios",
        "FINRA Markets": "Mutual fund data from Financial Industry Regulatory Authority",
        "Morningstar": "Investment research and financial data",
        "Thrift Savings Plan": "Federal employee retirement savings plan data",
        "IEX": "Investors Exchange - US stock market data",
        "Moscow Exchange": "Russian stock market data (MOEX)",
        "Nasdaq": "US stock exchange data and company information",
        "Naver Finance": "Korean stock market data (KOSPI/KOSDAQ)",
        "Tiingo": "Financial data provider with stocks, crypto, and fundamentals",
        "Quandl": "Financial and economic data (now Nasdaq Data Link)",
        "Alpha Vantage": "Real-time and historical stock/forex data",
        "Enigma": "Public data platform with various datasets",
        "Stooq": "Polish financial data provider",
        "Yahoo Options": "Options chain data from Yahoo Finance",
        "CSV Files": "Read data from CSV files",
        "Excel Files": "Read data from XLS/XLSX files",
        "JSON Files": "Read data from JSON files",
        "SQLite Database": "Read/write data from SQLite databases",
        "MongoDB": "Read/write data from MongoDB databases",
        "SQL Databases": "Read/write data from PostgreSQL, MySQL, etc."
    }


# Example usage and testing functions
def test_connections() -> Dict[str, bool]:
    """
    Test connections to various data sources

    Returns:
        Dictionary showing which sources are accessible
    """
    results = {}

    # Test each source with basic queries
    sources_to_test = {
        "Yahoo Finance": lambda: get_yahoo_data("AAPL", period="5d"),
        "FRED": lambda: get_fred_data("GDP", start_date="2020-01-01", end_date="2021-01-01"),
        # Add other tests as needed
    }

    for source, test_func in sources_to_test.items():
        try:
            test_func()
            results[source] = True
        except Exception:
            results[source] = False

    return results


if __name__ == "__main__":
    print("Data Access Module")
    print("Available sources:", list_available_sources())
    print("\nSource descriptions:")
    for source, desc in get_data_info().items():
        print(f"  {source}: {desc}")