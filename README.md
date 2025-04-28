# Momentum Stock Screener

A Python-based stock screener that identifies momentum breakouts and stocks approaching new 52-week highs with volume confirmation, enhanced with AI analysis capabilities.

## Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Core Algorithm](#core-algorithm)  
- [LLM Integration](#llm-integration)  
- [LLM Prompt System](#llm-prompt-system)  
- [LLM Output Structure](#llm-output-structure)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Custom Stock Lists](#custom-stock-lists)  
- [Performance Notes](#performance-notes)  
- [License](#license)  
- [Contributing](#contributing)  

## Overview

This tool scans various stock universes (S&P 500, S&P 1500, Russell indices, etc.) to identify stocks that are breaking out to new highs with strong volume. It uses a momentum-based approach to find potential trading opportunities based on proximity to 52-week highs and volume surges.

## Key Features

- Multiple stock universe options (S&P 500, S&P 1500, Russell 1000/3000, TSX, custom lists)  
- Robust data downloading with retry mechanisms and batch processing  
- Breakout detection based on proximity to 52-week highs and volume confirmation  
- AI-enhanced analysis using Google's Gemini API  
- Detailed statistics and verification of data quality  
- Excel export of results  

## Core Algorithm

1. **Average Volume Analysis**  
   - Calculate 50-day average volume for each stock (\( \overline{V}_{50} \)).  
   - Identify stocks with abnormally high current volume relative to their average.

2. **Rolling High Calculation**  
   - Determine the 52-week (or custom lookback period) rolling high for each stock.  
   - Focus on the highest price in the window to identify potential breakout levels.

3. **Volume Ratio Calculation**  
   - Compare today's volume \(V_{\text{today}}\) to the 50-day average volume \( \overline{V}_{50} \).  
   - Higher ratios indicate stronger interest in the stock.

4. **Proximity Calculation**  
   - Measure how close each stock is to its 52-week high:  
     \[
       \text{Proximity} = \frac{\text{High}_{52\text{wk}} - \text{Price}_{\text{today}}}{\text{High}_{52\text{wk}}}
     \]
   - Smaller proximity values indicate the stock is closer to breaking out to new highs.

5. **Breakout Classification**  
   - **High Breakers**: Stocks within 5% of their 52-week high **and** showing volume surge > 1.2× average.  
   - **Soft Breakouts**: Stocks within 5% of their 52-week high but without volume confirmation.

## LLM Integration

The screener integrates Google's Gemini 2.0 Flash model for enhanced analysis of screening results.

- **Model**: Google's Gemini 2.0 Flash  
- **Implementation**: Uses the `google-generativeai` Python package  
- **Authentication**: Requires a Gemini API key (stored in `.env` file)  

### LLM Functionality

The model performs several key functions:

- Validates screening parameters  
- Provides market context for the results  
- Analyzes patterns across breakout stocks  
- Offers suggestions for parameter refinements  
- Identifies potential anomalies in the data  

## LLM Prompt System

A structured system prompt guides the AI analysis through specific steps:

[VALIDATE]  Check parameters: universe_choice, soft_breakout_pct, proximity_threshold, volume_threshold, lookback_days.
[HYPOTHESIS] Explain your reasoning about what kind of stocks this screening should identify.
[CALCULATION] Explain key calculations (rolling high, proximity, volume ratio) in plain language.
[ANALYZE]  Interpret the results—what do the found breakouts indicate for trading?
[VERIFY]   Sanity-check outputs and highlight any anomalies.
[SUGGEST]  Recommend additional screening parameters that might improve results.

## LLM Output Structure

The output from the LLM is a JSON structure that contains the following fields:

- **Analysis**: Detailed analysis of the screening results
- **Recommendations**: Suggestions for improving the screening process
- **Anomalies**: Identified anomalies in the data

## Installation

To install the Momentum Stock Screener, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/momentum-stock-screener.git
   ```
2. Navigate to the project directory:
   ```
   cd momentum-stock-screener
   ```
3. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Set up the environment variables:
   - Create a `.env` file in the project root directory and add the following line:
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

## Configuration

The configuration for the Momentum Stock Screener is stored in the `config.py` file. You can customize the following parameters:

- **Stock Universe**: The list of stocks to be screened.
- **Lookback Period**: The time period used to calculate the 52-week high.
- **Proximity Threshold**: The percentage difference from the 52-week high to consider a stock as breaking out.
- **Volume Threshold**: The volume ratio to consider a stock as breaking out.

## Usage

To use the Momentum Stock Screener, run the following command:

```
python momentum_screener_llm.py
```

## Custom Stock Lists

You can specify custom stock lists in the `config.py` file. The format should be a list of stock symbols.

## Performance Notes

The Momentum Stock Screener is designed to be efficient and scalable. It can handle large datasets and provides detailed statistics and verification of data quality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

```mermaid
flowchart TB
  %%=== Subgraphs for logical layers ===%%
  subgraph CLI & Config
    direction TB
    U[User] --> |runs| Main[<b>momentum_screener_llm.py</b>]
    ENV[.env<br/>GEMINI_API_KEY] --- Main
  end

  subgraph Data_Fetching ["1. Data Fetching"]
    direction LR
    Main --> |calls| Fetcher[data_fetcher.py]
    Fetcher --> |yfinance / requests| RawData[Raw Price & Volume Data]
  end

  subgraph Processing ["2. Core Processing"]
    direction TB
    RawData --> VolCalc[VolumeCalculator<br/>(50-day avg)]
    RawData --> HighCalc[RollingHighCalculator<br/>(52-week high)]
    RawData --> ProxCalc[ProximityCalculator]
    RawData --> VolRatio[VolumeRatioCalculator]
    VolCalc & HighCalc & ProxCalc & VolRatio --> Screener[screener.py<br/>ScreenerEngine]
    Screener --> ScreenerResult[ScreenerResult JSON]
  end

  subgraph LLM_Analysis ["3. AI-Enhanced Analysis"]
    direction TB
    ScreenerResult --> LLMClient[llm_client.py]
    LLMClient --> |builds system prompt| Prompt[Structured Prompt]
    Prompt --> |API call| GeminiAPI[Gemini 2.0 Flash Model]
    GeminiAPI --> |returns| LLMRaw[LLM JSON Response]
    LLMRaw --> AnalysisProc[analysis.py<br/>AnalysisProcessor]
    AnalysisProc --> AIResult[Enhanced Analysis JSON]
  end

  subgraph Output ["4. Output & Export"]
    direction LR
    ScreenerResult --> Exporter[excel_exporter.py]
    AIResult --> Exporter
    Exporter --> |writes| ExcelFile[Excel Workbook]
    Main --> |prints| Console[Console Summary & Logs]
  end

  %%=== Styling ===%%
  classDef subgraphTitle fill:#efefef,stroke:#ccc,stroke-width:1px,font-weight:bold;
  class Data_Fetching,Processing,LLM_Analysis,Output subgraphTitle;
