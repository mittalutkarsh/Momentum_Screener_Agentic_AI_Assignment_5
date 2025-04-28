#!/usr/bin/env python3
"""
momentum_screener_quick.py - Quick Test Version of Momentum Screener

This version includes:
  - A quick test mode for faster testing with fewer tickers
  - Option to limit the number of tickers processed
  - Progress tracking with percentage complete
  - Reduced lookback period option for faster processing
"""
import os
import json
import asyncio
import time
import pickle
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import yfinance as yf
import requests
import bs4 as bs
import numpy as np

# Optional: Import LLM libraries if available
try:
    from dotenv import load_dotenv
    import google.generativeai as genai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Note: LLM integration not available. Install with: pip install google-generativeai python-dotenv")

# ---------- HELPER FUNCTIONS ----------
def get_major_tickers(count=30):
    """Get a list of major tickers for quick testing"""
    major_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "PG",
        "HD", "UNH", "MA", "AVGO", "COST", "MRK", "ABBV", "PEP", "ADBE", "KO",
        "CSCO", "TMO", "MCD", "CRM", "ACN", "NFLX", "BAC", "AMD", "CMCSA", "ORCL",
        "PFE", "INTC", "DIS", "IBM", "QCOM", "CAT", "GE", "NKE", "VZ", "XOM"
    ]
    return major_tickers[:count]

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

def download_data_in_batches(tickers, start_date, end_date, batch_size=20, progress=True):
    """Download data in batches with progress tracking"""
    all_data = {}
    num_batches = len(tickers) // batch_size + (1 if len(tickers) % batch_size != 0 else 0)
    
    total_count = len(tickers)
    processed_count = 0
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(tickers))
        batch_tickers = tickers[batch_start:batch_end]
        
        processed_count += len(batch_tickers)
        percent_complete = round((processed_count / total_count) * 100)
        
        if progress:
            print(f"\rDownloading: {percent_complete}% complete ({processed_count}/{total_count} tickers)", end="")
        
        # Try up to 2 times with short backoff (faster for testing)
        for attempt in range(2):
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
                if attempt < 1:  # Only wait once
                    time.sleep(1)  # Simple 1-second retry
    
    if progress:
        print("\rDownload complete                                           ")
    
    return all_data

def quick_momentum_screener(
    tickers=None,
    ticker_count=30,
    soft_breakout_pct=0.005,
    proximity_threshold=0.05,
    volume_threshold=1.2,
    lookback_days=90
):
    """Quick version of momentum screener for testing"""
    # Use provided tickers or get major tickers
    if tickers is None:
        universe = get_major_tickers(ticker_count)
        universe_name = f"Quick Test ({ticker_count} major tickers)"
    else:
        universe = clean_tickers(tickers[:ticker_count])
        universe_name = f"Custom ({len(universe)} tickers)"
    
    print(f"\nUniverse Loaded: {universe_name}")
    
    # Set date range (shorter lookback for speed)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Download data with progress indicator
    data_dict = download_data_in_batches(universe, start_date, end_date, batch_size=10)
    
    # Clean and prepare data
    universe_cleaned = list(data_dict.keys())
    print(f"Successfully downloaded: {len(universe_cleaned)} tickers")
    
    missing_tickers = list(set(universe) - set(universe_cleaned))
    if missing_tickers:
        print(f"Missing tickers: {missing_tickers}")
    
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
            try:
                close_prices[ticker] = ticker_data[('Close', ticker)] if ('Close', ticker) in ticker_data.columns else ticker_data['Close']
                volumes[ticker] = ticker_data[('Volume', ticker)] if ('Volume', ticker) in ticker_data.columns else ticker_data['Volume']
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
    
    # Check if we have valid data
    if close_prices.empty or volumes.empty:
        return {
            "error": "No valid price/volume data found"
        }
    
    # Calculate rolling highs and volume averages (using shorter window for faster processing)
    rolling_high = close_prices.rolling(lookback_days, min_periods=1).max()
    avg_vol_window = min(30, lookback_days // 3)  # Use shorter window for volume average
    avg_vol = volumes.rolling(avg_vol_window, min_periods=5).mean()
    
    # Get today's values
    current_close = close_prices.iloc[-1]
    rolling_high_today = rolling_high.iloc[-1]
    latest_volume = volumes.iloc[-1]
    latest_avg_volume = avg_vol.iloc[-1]
    
    # Calculate volume ratio and proximity
    volume_ratio = latest_volume / latest_avg_volume
    proximity = (rolling_high_today - current_close) / rolling_high_today
    
    # Identify breakouts and near breakouts
    breakouts = []
    near_breakouts = []
    
    for ticker in universe_cleaned:
        if pd.isna(proximity[ticker]) or pd.isna(volume_ratio[ticker]):
            continue
        
        ticker_data = {
            "Symbol": ticker,
            "Price": round(float(current_close[ticker]), 2),
            "High": round(float(rolling_high_today[ticker]), 2),
            "Distance": round(float(proximity[ticker] * 100), 2),
            "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
        }
        
        # Classify as breakout or near breakout
        if proximity[ticker] <= soft_breakout_pct and volume_ratio[ticker] > volume_threshold:
            breakouts.append(ticker_data)
        elif proximity[ticker] <= proximity_threshold:
            near_breakouts.append(ticker_data)
    
    # Sort by proximity to high
    breakouts.sort(key=lambda x: x["Distance"])
    near_breakouts.sort(key=lambda x: x["Distance"])
    
    # Export to Excel
    today = datetime.today().strftime('%Y-%m-%d')
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/QuickTest_Momentum_Screener_{today}.xlsx"
    
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(breakouts).to_excel(writer, sheet_name="Breakouts", index=False)
        pd.DataFrame(near_breakouts).to_excel(writer, sheet_name="Near Breakouts", index=False)
    
    print(f"Excel saved to: {output_path}")
    
    # Output Summary
    print("\n=== Summary ===")
    print(f"Universe: {universe_name} | Date: {today}")
    print(f"Breakouts Found: {len(breakouts)}")
    print(f"Near Breakouts Found: {len(near_breakouts)}")
    
    # Print breakouts
    if breakouts:
        print("\nBreakouts:")
        for b in breakouts:
            print(f"  {b['Symbol']}: ${b['Price']} ({b['Distance']}% from high, {b['Volume_Ratio']}x volume)")
    
    # Print near breakouts (limit to 5 for brevity)
    if near_breakouts:
        print("\nNear Breakouts (top 5):")
        for b in near_breakouts[:5]:
            print(f"  {b['Symbol']}: ${b['Price']} ({b['Distance']}% from high, {b['Volume_Ratio']}x volume)")
    
    return {
        "breakouts": breakouts,
        "near_breakouts": near_breakouts,
        "excel_path": output_path
    }

if __name__ == "__main__":
    print("=== Quick Test Momentum Screener ===")
    print("This is a faster version for testing with fewer tickers and shorter lookback period.")
    
    # Get quick test parameters
    try:
        ticker_count = int(input("\nNumber of tickers to test [5-50, default=30]: ") or "30")
        ticker_count = max(5, min(50, ticker_count))  # Limit between 5 and 50
        
        lookback_days = int(input("Lookback days [30-365, default=90]: ") or "90")
        lookback_days = max(30, min(365, lookback_days))  # Limit between 30 and 365
        
        # Run quick test
        result = quick_momentum_screener(
            ticker_count=ticker_count,
            lookback_days=lookback_days
        )
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"\nError: {e}")