#!/usr/bin/env python3
"""
momentum_screener_llm.py - Enhanced Momentum Screener with LLM Integration

Features:
  - Robust stock data downloading with batch processing and error handling
  - Multiple universe options (S&P 500, S&P 1500, Russell indexes, TSX, custom)
  - LLM-enhanced reasoning and reporting for momentum screening
  - Breakout detection based on proximity to highs and volume surge
  - Excel export with detailed analysis

Dependencies:
  pandas, yfinance, requests, bs4, numpy, google-generativeai, python-dotenv
Install:
  pip install pandas yfinance requests beautifulsoup4 numpy google-generativeai python-dotenv
"""
import os
import json
import asyncio
import time
import pickle
import random
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import re

import pandas as pd
import yfinance as yf
import requests
import bs4 as bs
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('momentum_screener')

# ---------- LLM CONFIGURATION ----------
def load_llm_config():
    """Load environment variables and configure Gemini API"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    logger.info(f"API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        logger.info(f"API Key starts with: {api_key[:5]}...")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env file. Please add it.")
    
    genai.configure(api_key=api_key)
    # Use the correct model name with the models/ prefix
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    logger.info("Gemini model configured successfully")
    return model

# System prompt for the LLM reasoning framework
SYSTEM_PROMPT = """You are a Momentum Screener agent with deep expertise in technical analysis and stock screening. Follow this framework:

[VALIDATE] Check parameters: universe_choice, soft_breakout_pct(0.001-0.02), proximity_threshold(0.01-0.1), volume_threshold(1.0-3.0), lookback_days(90-365).
[HYPOTHESIS] Explain your reasoning about what kind of stocks this screening should identify.
[CALCULATION] Explain key calculations (rolling high, proximity, volume ratio) in plain language.
[ANALYZE] Interpret the results - what do the found breakouts indicate for trading?
[VERIFY] Sanity-check outputs and highlight any anomalies.
[SUGGEST] Recommend 1-3 additional screening parameters that might improve results.

IMPORTANT: Your response must be ONLY valid JSON with no narrative text before or after.

OUTPUT FORMAT: Return a JSON object with these keys:
  "reasoning": {
    "parameter_check": "Explanation of parameter validity",
    "strategy_summary": "What this momentum strategy aims to capture",
    "calculation_explanation": "Key metrics explained",
    "verification": "Quality check of results"
  },
  "analysis": {
    "market_context": "Brief market context for these results",
    "top_breakouts_analysis": "Analysis of the most promising breakouts",
    "pattern_recognition": "Any patterns across the breakout stocks"
  },
  "suggestions": {
    "parameter_adjustments": "Suggestions for tweaking parameters",
    "additional_filters": "Other metrics to consider adding"
  }
}
"""

# ---------- HELPER FUNCTIONS ----------
@lru_cache(maxsize=5)
def save_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia and return the list"""
    try:
        logger.info("Scraping S&P 500 tickers from Wikipedia")
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        
        # Try to find the table - more verbose to help debug
        tables = soup.find_all('table', {'class': 'wikitable'})
        logger.info(f"Found {len(tables)} tables with class 'wikitable'")
        
        if not tables:
            raise ValueError("No tables with class 'wikitable' found")
        
        # Use the first table with class 'wikitable'
        table = tables[0]
        
        tickers = []
        # Find all rows except the header
        rows = table.find_all('tr')[1:]  # Skip header row
        logger.info(f"Found {len(rows)} rows in table")
        
        for row in rows:
            # Get all cells in the row
            cells = row.find_all('td')
            if cells:  # Make sure there are cells in the row
                # The ticker symbol should be in the first column
                ticker = cells[0].text.strip()
                if ticker:  # Only add non-empty tickers
                    tickers.append(ticker)
        
        if not tickers:
            raise ValueError("No tickers found in the table")
        
        logger.info(f"Successfully scraped {len(tickers)} tickers")
        
        # Save to pickle for future use
        with open("sp500tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)
        
        return tickers
    except Exception as e:
        logger.error(f"Error scraping S&P 500 tickers: {e}")
        
        # Try alternative method: direct list of most S&P 500 stocks
        try:
            # This is a more comprehensive fallback list
            major_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG", 
                            "UNH", "XOM", "JNJ", "WMT", "MA", "LLY", "CVX", "HD", "AVGO", "MRK", 
                            "PEP", "COST", "ABBV", "KO", "BAC", "PFE", "TMO", "CSCO", "MCD", "CRM", 
                            "ABT", "DHR", "ACN", "ADBE", "WFC", "DIS", "AMD", "CMCSA", "TXN", "NEE",
                            "VZ", "PM", "INTC", "NFLX", "RTX", "QCOM", "IBM", "ORCL", "HON", "BMY"]
            
            logger.warning(f"Using extended fallback list of {len(major_tickers)} major tickers")
            return major_tickers
        except Exception:
            # Last resort - use the original small set
            logger.warning("Using small fallback set of 10 major tickers")
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG"]

def clean_tickers(tickers):
    """Clean and normalize ticker symbols"""
    cleaned_tickers = []
    for ticker in tickers:
        # Remove newlines and whitespace
        ticker = ticker.strip()
        
        # Handle special cases
        if ticker == 'BF.B':
            ticker = 'BF-B'
        elif '.' in ticker:
            ticker = ticker.replace('.', '-')
            
        cleaned_tickers.append(ticker)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(cleaned_tickers))

def download_data_in_batches(tickers, start_date, end_date, batch_size=100):
    """Download data in batches to avoid timeouts"""
    all_data = {}
    num_batches = len(tickers) // batch_size + (1 if len(tickers) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(tickers))
        batch_tickers = tickers[batch_start:batch_end]
        
        logger.info(f"\nDownloading batch {i+1}/{num_batches} ({batch_start+1}-{batch_end} of {len(tickers)} tickers)")
        
        # Try up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                batch_data = yf.download(
                    batch_tickers, 
                    start=start_date, 
                    end=end_date, 
                    group_by="ticker", 
                    auto_adjust=True,
                    progress=False
                )
                
                # Process successfully downloaded tickers
                if isinstance(batch_data, pd.DataFrame) and len(batch_tickers) == 1:
                    # Special case for single ticker (different structure)
                    all_data[batch_tickers[0]] = batch_data
                else:
                    # For multiple tickers
                    for ticker in batch_tickers:
                        if ticker in batch_data.columns.levels[0]:
                            all_data[ticker] = batch_data[ticker]
                
                break  # Success, exit retry loop
                
            except Exception as e:
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(f"  Batch download failed (attempt {attempt+1}/3). Error: {str(e)}")
                if attempt < 2:  # Don't wait after the last attempt
                    logger.info(f"  Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
    
    return all_data

def get_stock_universe(universe_choice):
    """Load stock universe based on user choice"""
    if universe_choice == 0:
        universe_name = "SPY"
        tickers = save_sp500_tickers()
        universe = clean_tickers(tickers)
    elif universe_choice == 1:
        universe_name = "S&P1500"
        try:
            # Try to use our own scraper first for the S&P 500
            sp500_tickers = save_sp500_tickers()
            
            # Use pd.read_html for S&P 400 and 600
            df2 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]
            df3 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]
            
            midcap_tickers = df2['Symbol'].tolist()
            smallcap_tickers = df3['Symbol'].tolist()
            
            universe = clean_tickers(sp500_tickers + midcap_tickers + smallcap_tickers)
        except Exception as e:
            logger.error(f"Error loading S&P1500: {e}")
            logger.warning("Falling back to S&P 500 only")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 2:
        universe_name = "Russell 1000"
        try:
            universe = clean_tickers(pd.read_csv("russell1000.csv")['Symbol'].tolist())
        except FileNotFoundError:
            logger.warning("russell1000.csv not found. Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 3:
        universe_name = "Russell 3000"
        try:
            universe = clean_tickers(pd.read_csv("russell3000.csv")['Symbol'].tolist())
        except FileNotFoundError:
            logger.warning("russell3000.csv not found. Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 4:
        universe_name = "TSX Composite"
        try:
            df = pd.read_html('https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index')[0]
            universe = clean_tickers(df['Symbol'].tolist())
        except Exception as e:
            logger.error(f"Error loading TSX Composite: {e}")
            logger.warning("Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 5:
        custom_file = input("Enter path to text file with one ticker per line: ")
        universe_name = f"Custom ({os.path.basename(custom_file)})"
        try:
            with open(custom_file, 'r') as f:
                universe = clean_tickers([line.strip() for line in f.readlines()])
        except FileNotFoundError:
            logger.warning(f"File {custom_file} not found. Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    else:
        logger.warning("Invalid choice. Using S&P 500")
        universe = clean_tickers(save_sp500_tickers())
        universe_name = "SPY (fallback)"
    
    return universe, universe_name

# ---------- CORE FUNCTIONALITY ----------
def momentum_screener(
    universe_choice=0,
    soft_breakout_pct=0.005,
    proximity_threshold=0.05,
    volume_threshold=1.2,
    lookback_days=365
):
    """Execute momentum screening strategy"""
    # Get stock universe
    universe, universe_name = get_stock_universe(universe_choice)
    logger.info(f"\nUniverse Loaded: {universe_name} ({len(universe)} tickers)")
    
    # Set date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    logger.info(f"\nDownloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Download data
    data_dict = download_data_in_batches(universe, start_date, end_date)
    
    # Clean and prepare data
    universe_cleaned = list(data_dict.keys())
    
    # Download statistics
    logger.info(f"\n=== Download Statistics ===")
    logger.info(f"Universe selected: {universe_name}")
    logger.info(f"Total tickers requested: {len(universe)}")
    logger.info(f"Successfully downloaded: {len(universe_cleaned)} ({round(len(universe_cleaned)/len(universe)*100, 1)}%)")
    
    missing_tickers = list(set(universe) - set(universe_cleaned))
    logger.info(f"Missing tickers: {len(missing_tickers)} ({round(len(missing_tickers)/len(universe)*100, 1)}%)")
    
    if len(universe_cleaned) < 0.8 * len(universe):
        logger.warning("\n⚠️ WARNING: Less than 80% of tickers successfully downloaded!")
        logger.warning("Consider checking your internet connection or using a smaller universe.")
    
    if missing_tickers:
        logger.info(f"Missing tickers skipped: {len(missing_tickers)}")
        if len(missing_tickers) <= 20:
            logger.info(f"Missing: {missing_tickers}")
        else:
            logger.info(f"First 20 missing: {missing_tickers[:20]}...")
    
    # Extract close prices and volumes into dataframes
    close_prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for ticker in universe_cleaned:
        ticker_data = data_dict[ticker]
        
        # Handle different data structures
        if 'Close' in ticker_data.columns:
            close_prices[ticker] = ticker_data['Close']
            volumes[ticker] = ticker_data['Volume']
        else:
            # In case yfinance returns data with MultiIndex columns
            try:
                close_prices[ticker] = ticker_data[('Close', ticker)] if ('Close', ticker) in ticker_data.columns else ticker_data['Close']
                volumes[ticker] = ticker_data[('Volume', ticker)] if ('Volume', ticker) in ticker_data.columns else ticker_data['Volume']
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
    
    # Data quality verification
    empty_price_count = (close_prices.isna().sum() > close_prices.shape[0] * 0.9).sum()
    if empty_price_count > 0:
        logger.warning(f"\n⚠️ WARNING: {empty_price_count} tickers have more than 90% missing price data")

    # Recent data check - ensure we have recent data
    if not close_prices.empty:
        latest_date = close_prices.index[-1].strftime('%Y-%m-%d')
        today = datetime.today().strftime('%Y-%m-%d')
        days_diff = (datetime.today() - close_prices.index[-1]).days

        if days_diff > 5:
            logger.warning(f"\n⚠️ WARNING: Most recent data is from {latest_date}, which is {days_diff} days old")
            logger.warning("This may be due to market holidays or data availability issues")
    
    # Check if we have valid data
    if close_prices.empty or volumes.empty:
        logger.error("No valid price/volume data found")
        return {
            "error": {
                "type": "DataError",
                "description": "No valid price/volume data found",
                "suggestion": "Try a different universe or check internet connection"
            }
        }
    
    # Calculate rolling highs and volume averages
    rolling_high = close_prices.rolling(lookback_days, min_periods=1).max()
    avg_vol_50 = volumes.rolling(50, min_periods=5).mean()
    
    # Get today's values
    current_close = close_prices.iloc[-1]
    rolling_high_today = rolling_high.iloc[-1]
    latest_volume = volumes.iloc[-1]
    latest_avg_volume = avg_vol_50.iloc[-1]
    
    # Calculate volume ratio and proximity
    volume_ratio = latest_volume / latest_avg_volume
    proximity = (rolling_high_today - current_close) / rolling_high_today
    
    # Sample verification (random 5 tickers)
    if len(universe_cleaned) > 5:
        sample_tickers = random.sample(universe_cleaned, 5)
        logger.info("\n=== Sample Data Verification ===")
        for ticker in sample_tickers:
            logger.info(f"{ticker}: {len(data_dict[ticker])} days of data, latest price: ${current_close[ticker]:.2f}")
            logger.info(f"  52-week high: ${rolling_high_today[ticker]:.2f}, distance: {proximity[ticker]*100:.2f}%")
            logger.info(f"  Volume ratio: {volume_ratio[ticker]:.2f}x")
    
    # Identify breakouts and near breakouts
    high_breakers = (proximity <= soft_breakout_pct) & (volume_ratio > volume_threshold)
    near_highs = (proximity <= proximity_threshold) & (~high_breakers) & (rolling_high_today > 0)
    
    # Convert to series with index for easier filtering
    high_breakers = pd.Series(high_breakers, index=universe_cleaned).fillna(False)
    near_highs = pd.Series(near_highs, index=universe_cleaned).fillna(False)
    
    # Debugging output for breakout conditions
    logger.info("\n=== Breakout Condition Stats ===")
    close_to_high = (proximity <= proximity_threshold).sum()
    high_volume = (volume_ratio > volume_threshold).sum()
    soft_breakouts = (proximity <= soft_breakout_pct).sum()

    logger.info(f"Tickers close to high ({proximity_threshold*100}% threshold): {close_to_high}")
    logger.info(f"Tickers with high volume ({volume_threshold}x threshold): {high_volume}")
    logger.info(f"Tickers at breakout level ({soft_breakout_pct*100}% threshold): {soft_breakouts}")
    logger.info(f"Breakouts with volume confirmation: {high_breakers.sum()}")
    
    # Create results lists
    breakouts = []
    near_breakouts = []
    
    # Populate breakouts list
    for ticker in high_breakers[high_breakers].index:
        breakouts.append({
            "Symbol": ticker,
            "Price": round(float(current_close[ticker]), 2),
            "52_Week_High": round(float(rolling_high_today[ticker]), 2),
            "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
            "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
        })
    
    # Populate near breakouts list
    for ticker in near_highs[near_highs].index:
        near_breakouts.append({
            "Symbol": ticker,
            "Price": round(float(current_close[ticker]), 2),
            "52_Week_High": round(float(rolling_high_today[ticker]), 2),
            "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
            "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
        })
    
    # Sort by proximity to high
    breakouts.sort(key=lambda x: x["Distance_to_High_pct"])
    near_breakouts.sort(key=lambda x: x["Distance_to_High_pct"])
    
    # Export to Excel
    today = datetime.today().strftime('%Y-%m-%d')
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{universe_name}_Momentum_Screener_{today}.xlsx"
    
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(breakouts).to_excel(writer, sheet_name="Breakouts", index=False)
        pd.DataFrame(near_breakouts).to_excel(writer, sheet_name="Near Breakouts", index=False)
    
    logger.info(f"\nExcel saved to: {output_path}")
    
    # Output Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Universe: {universe_name} | Date: {today}")
    logger.info(f"Breakouts Found: {len(breakouts)}")
    logger.info(f"Near Breakouts Found: {len(near_breakouts)}")
    
    # Return results
    return {
        "breakouts": breakouts,
        "near_breakouts": near_breakouts,
        "excel_path": output_path,
        "parameters": {
            "universe": universe_name,
            "soft_breakout_pct": soft_breakout_pct,
            "proximity_threshold": proximity_threshold,
            "volume_threshold": volume_threshold,
            "lookback_days": lookback_days
        }
    }

async def generate_momentum_analysis(result_data):
    """Generate LLM analysis for the momentum screening results"""
    # DEBUG
    logger.info("=" * 50)
    logger.info("ENTERING LLM ANALYSIS FUNCTION")
    logger.info(f"Using API key: {os.getenv('GEMINI_API_KEY')[:5]}..." if os.getenv('GEMINI_API_KEY') else "NO API KEY FOUND")
    logger.info("=" * 50)
    
    # Skip LLM if there's an error in the results
    if "error" in result_data:
        logger.info("Skipping LLM due to error in results")
        return result_data
    
    # Initialize LLM model
    try:
        logger.info("Initializing LLM model...")
        model = load_llm_config()
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return {
            "reasoning": {
                "parameter_check": f"LLM unavailable: {str(e)}",
                "strategy_summary": "Standard momentum breakout strategy",
                "verification": f"Found {len(result_data['breakouts'])} breakouts and {len(result_data['near_breakouts'])} near-breakouts"
            },
            "result": result_data
        }
    
    # Prepare prompt for the LLM
    logger.info("Preparing prompt for LLM...")
    prompt = f"""
{SYSTEM_PROMPT}

Analyze these momentum screening results:
{json.dumps(result_data, indent=2)}
"""
    logger.info(f"Prompt length: {len(prompt)} characters")
    
    # Call the LLM with timeout
    try:
        logger.info("Calling Gemini API...")
        resp = await asyncio.wait_for(model.generate_content_async(prompt), timeout=30)
        logger.info(f"Response received, length: {len(resp.text)}")
        
        # Process the response - handle markdown code blocks
        json_text = resp.text
        # Remove markdown code block if present
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        
        logger.info(f"JSON text after cleaning markdown: {len(json_text)} characters")
        
        try:
            llm_analysis = json.loads(json_text)
            logger.info("Successfully parsed JSON response")
            
            # Ensure all required fields exist
            llm_analysis.setdefault("reasoning", {})
            llm_analysis.setdefault("analysis", {})
            llm_analysis.setdefault("suggestions", {})
            
            return {
                "reasoning": llm_analysis.get("reasoning", {}),
                "analysis": llm_analysis.get("analysis", {}),
                "suggestions": llm_analysis.get("suggestions", {}),
                "result": result_data
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"First 100 chars of response: {resp.text[:100]}...")
            
            # Try to extract JSON from text
            try:
                # Look for JSON-like structure between curly braces
                json_match = re.search(r'(\{.*\})', resp.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    llm_analysis = json.loads(json_str)
                    logger.info("Successfully extracted and parsed JSON from response")
                    # Ensure all required fields exist
                    llm_analysis.setdefault("reasoning", {})
                    llm_analysis.setdefault("analysis", {})
                    llm_analysis.setdefault("suggestions", {})
                    return {
                        "reasoning": llm_analysis.get("reasoning", {}),
                        "analysis": llm_analysis.get("analysis", {}),
                        "suggestions": llm_analysis.get("suggestions", {}),
                        "result": result_data
                    }
                else:
                    raise ValueError("No JSON structure found in response")
            except Exception as e:
                logger.error(f"JSON extraction failed: {e}")
                # Create a minimal response
                return {
                    "reasoning": {
                        "parameter_check": "LLM response parsing failed",
                        "strategy_summary": "Standard momentum breakout strategy",
                        "verification": f"Found {len(result_data['breakouts'])} breakouts and {len(result_data['near_breakouts'])} near-breakouts"
                    },
                    "result": result_data
                }
    except asyncio.TimeoutError:
        logger.error("LLM analysis timed out")
        return {
            "reasoning": {
                "parameter_check": "LLM analysis timed out - using local analysis only",
                "strategy_summary": "Standard momentum breakout strategy",
                "verification": f"Found {len(result_data['breakouts'])} breakouts and {len(result_data['near_breakouts'])} near-breakouts"
            },
            "result": result_data
        }
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return {
            "reasoning": {
                "parameter_check": f"LLM error: {str(e)}",
                "strategy_summary": "Standard momentum breakout strategy",
                "verification": f"Found {len(result_data['breakouts'])} breakouts and {len(result_data['near_breakouts'])} near-breakouts"
            },
            "result": result_data
        }

async def run_momentum_screener(
    universe_choice=0,
    soft_breakout_pct=0.005,
    proximity_threshold=0.05,
    volume_threshold=1.2,
    lookback_days=365,
    use_llm=True
):
    """Main function to run the momentum screener with optional LLM analysis"""
    # Run the screener
    result = momentum_screener(
        universe_choice, 
        soft_breakout_pct,
        proximity_threshold,
        volume_threshold,
        lookback_days
    )
    
    # Add LLM analysis if requested, even if no breakouts found
    if use_llm:
        # Force LLM analysis regardless of results
        logger.info("Running LLM analysis...") # Debug print
        return await generate_momentum_analysis(result)
    else:
        return {
            "reasoning": {
                "parameter_check": "Using provided parameters directly",
                "strategy_summary": "Standard momentum breakout strategy",
                "verification": f"Found {len(result.get('breakouts', []))} breakouts and {len(result.get('near_breakouts', []))} near-breakouts"
            },
            "result": result
        }

# ---------- COMMAND LINE INTERFACE ----------
def print_analysis_report(analysis_data):
    """Print formatted analysis report to console"""
    if "error" in analysis_data.get("result", {}):
        print("\n⚠️ ERROR ⚠️")
        error = analysis_data["result"]["error"]
        print(f"Type: {error['type']}")
        print(f"Description: {error['description']}")
        print(f"Suggestion: {error['suggestion']}")
        return
    
    print("\n=== MOMENTUM SCREENER ANALYSIS ===")
    
    if "reasoning" in analysis_data:
        print("\n🔍 STRATEGY SUMMARY")
        print(analysis_data["reasoning"].get("strategy_summary", "Standard momentum breakout strategy"))
        
        if "calculation_explanation" in analysis_data["reasoning"]:
            print("\n📊 CALCULATION METHOD")
            print(analysis_data["reasoning"]["calculation_explanation"])
    
    if "analysis" in analysis_data:
        print("\n📈 MARKET CONTEXT")
        print(analysis_data["analysis"].get("market_context", "No market context available"))
        
        if "top_breakouts_analysis" in analysis_data["analysis"]:
            print("\n🔝 TOP BREAKOUTS ANALYSIS")
            print(analysis_data["analysis"]["top_breakouts_analysis"])
    
    if "suggestions" in analysis_data:
        print("\n💡 SUGGESTIONS")
        if "parameter_adjustments" in analysis_data["suggestions"]:
            print("Parameter Adjustments:", analysis_data["suggestions"]["parameter_adjustments"])
        if "additional_filters" in analysis_data["suggestions"]:
            print("Additional Filters:", analysis_data["suggestions"]["additional_filters"])
    
    result = analysis_data["result"]
    
    print("\n=== BREAKOUTS ===")
    if not result["breakouts"]:
        print("None found")
    else:
        # Print the first 5 breakouts
        print(f"Found {len(result['breakouts'])} breakouts. Top 5:")
        for i, b in enumerate(result["breakouts"][:5], 1):
            print(f"{i}. {b['Symbol']}: ${b['Price']} ({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)")
    
    print("\n=== NEAR BREAKOUTS ===")
    if not result["near_breakouts"]:
        print("None found")
    else:
        # Print the first 5 near breakouts
        print(f"Found {len(result['near_breakouts'])} near breakouts. Top 5:")
        for i, b in enumerate(result["near_breakouts"][:5], 1):
            print(f"{i}. {b['Symbol']}: ${b['Price']} ({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)")
    
    print(f"\nFull results saved to Excel: {result['excel_path']}")

async def main():
    """Main entry point with CLI interface"""
    print("=== LLM-Enhanced Momentum Screener ===")
    print("This tool identifies stocks breaking out to new highs with volume confirmation")
    print("\n=== Universe Options ===")
    print("0 - SPY (S&P 500)")
    print("1 - S&P1500")
    print("2 - Russell 1000 (CSV required)")
    print("3 - Russell 3000 (CSV required)")
    print("4 - TSX Composite")
    print("5 - Custom (from text file)")
    
    try:
        # Get user inputs with defaults
        universe_choice = int(input("\nSelect Universe [0-5, default=0]: ") or "0")
        soft_breakout_pct = float(input("Soft Breakout Percentage [0.001-0.02, default=0.005]: ") or "0.005")
        proximity_threshold = float(input("Proximity Threshold [0.01-0.1, default=0.05]: ") or "0.05")
        volume_threshold = float(input("Volume Threshold [1.0-3.0, default=1.2]: ") or "1.2")
        lookback_days = int(input("Lookback Days [90-365, default=365]: ") or "365")
        use_llm_input = input("Use LLM for enhanced analysis? [y/N, default=N]: ").lower()
        use_llm = use_llm_input == 'y' or use_llm_input == 'yes'
        
        # Validate inputs
        if not (0 <= universe_choice <= 5):
            universe_choice = 0
            print("Invalid universe choice. Using S&P 500.")
        
        if not (0.001 <= soft_breakout_pct <= 0.02):
            soft_breakout_pct = 0.005
            print("Invalid soft breakout percentage. Using 0.005.")
        
        if not (0.01 <= proximity_threshold <= 0.1):
            proximity_threshold = 0.05
            print("Invalid proximity threshold. Using 0.05.")
        
        if not (1.0 <= volume_threshold <= 3.0):
            volume_threshold = 1.2
            print("Invalid volume threshold. Using 1.2.")
        
        if not (90 <= lookback_days <= 365):
            lookback_days = 365
            print("Invalid lookback days. Using 365.")
        
        # Run the screener
        analysis = await run_momentum_screener(
            universe_choice,
            soft_breakout_pct,
            proximity_threshold,
            volume_threshold,
            lookback_days,
            use_llm
        )
        
        # Print the results
        print_analysis_report(analysis)
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())