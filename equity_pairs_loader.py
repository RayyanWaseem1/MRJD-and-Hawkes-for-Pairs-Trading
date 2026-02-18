"""
Data pipeline for loading equity pairs (NVDA and AMD)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EquityPairsDataPipeline:
    """
    Data preprocessing for equity pairs (NVDA and AMD)

    """

    def __init__(self, asset_a_path: Optional[str] = None, asset_b_path: Optional[str] = None):
        """
        Initialize data pipeline

        Params:
        - asset_a_path: str
            -path to first asset CSV
        - asset_b_path: str
            -path to second asset CSV

        """
        self.asset_a_path = asset_a_path
        self.asset_b_path = asset_b_path
        self.data = {}

    def load_from_csv(self,
                      asset_a_path: Optional[str] = None,
                      asset_b_path: Optional[str] = None,
                      date_columns: str = 'Date',
                      parse_dates: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Params:
        - asset_a_path: str
            - path to asset a CSV
        - asset_b_path: str
            - path to asset b CSV
        -date_column: str
            -name of date column in CSV
        -parse_dates: bool
            -whether to parse dates 

        Returns:
        - Dict[str, pd.DataFrame]
            - dictionary with keys 'asset_a' and 'asset_b' containing loaded dataframes
        """
        asset_a_path = asset_a_path or self.asset_a_path
        asset_b_path = asset_b_path or self.asset_b_path

        if asset_a_path is None or asset_b_path is None:
            raise ValueError("Both asset_a_path and asset_b_path must be provided")

        print(f"Loading equity pairs data from the CSV files...")
        print(f" Asset A (NVDA): {asset_a_path}")
        print(f" Asset B (AMD): {asset_b_path}")

        try:
            #loading Asset A (NVDA)
            asset_a = pd.read_csv(asset_a_path)

            #handling date column
            date_col_a = date_columns if date_columns in asset_a.columns else None
            if date_col_a is None:
                cols_lower_a = {str(col).lower(): col for col in asset_a.columns}
                date_col_a = (
                    cols_lower_a.get(str(date_columns).lower())
                    or cols_lower_a.get('ts_event')
                    or cols_lower_a.get('date')
                )

            if date_col_a is not None:
                if parse_dates:
                    asset_a[date_col_a] = pd.to_datetime(asset_a[date_col_a], errors='coerce')
                asset_a = asset_a.set_index(date_col_a)
            else:
                print(f"Warning: No date column found for Asset A; using integer index")

            #ensure proper column names (case-insensitive)
            asset_a.columns = [col.capitalize() for col in asset_a.columns]

            #loading Asset B (AMD)
            asset_b = pd.read_csv(asset_b_path)

            #handling date column
            date_col_b = date_columns if date_columns in asset_b.columns else None
            if date_col_b is None:
                cols_lower_b = {str(col).lower(): col for col in asset_b.columns}
                date_col_b = (
                    cols_lower_b.get(str(date_columns).lower())
                    or cols_lower_b.get('ts_event')
                    or cols_lower_b.get('date')
                )

            if date_col_b is not None:
                if parse_dates:
                    asset_b[date_col_b] = pd.to_datetime(asset_b[date_col_b], errors='coerce')
                asset_b = asset_b.set_index(date_col_b)
            else:
                print(f"Warning: No date column found for Asset B; using integer index")

            asset_b.columns = [col.capitalize() for col in asset_b.columns]

            #validate the columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in asset_a.columns:
                    raise ValueError(f"Missing columns '{col}' in Asset A date")
                if col not in asset_b.columns:
                    raise ValueError(f"Missing columns '{col}' in Asset B data")
                
            self.data = {
                'asset_a': asset_a,
                'asset_b': asset_b
            }

            print("Successfully loaded data")
            print(f" Asset A (NVDA): {len(asset_a)} observations")
            print(f" Asset B (AMD): {len(asset_b)} observations")
            print(f" Date range: {asset_a.index[0]} to {asset_a.index[-1]}")

            return self.data
        except FileNotFoundError as e:
            print(f"Error: CSV file not found - {str(e)}")
            print("Please check the file paths and try again.")
            raise 

        except Exception as e:
            print(f"Error loading CSV files: {str(e)}")
            raise 

    def clean_data(self) -> Dict[str, pd.DataFrame]:
        """
        Cleaning data
        -removing missing values
        -handling outliers
        -align timestamps

        Returns:
        -Dict[str, pd.DataFrame]
            -cleaned data for asset_a and asset_b
        """
        print("Cleaning data...")

        if not self.data:
            raise ValueError("No data to clean. Run load_from_csv() first.")
        
        #Aligning timestamps
        common_index = self.data['asset_a'].index.intersection(self.data['asset_b'].index)

        cleaned_data = {
            'asset_a': self.data['asset_a'].loc[common_index].copy(),
            'asset_b': self.data['asset_b'].loc[common_index].copy()
        }

        #Removing missing values
        for key in cleaned_data:
            cleaned_data[key] = cleaned_data[key].dropna()

        #Removing outliers (prices that change more than 50% in one day)
        for key in cleaned_data:
            returns = cleaned_data[key]['Close'].pct_change()
            mask = (returns.abs() < 0.5)
            cleaned_data[key] = cleaned_data[key][mask]

        #Re-align after outlier removal 
        common_index = cleaned_data['asset_a'].index.intersection(cleaned_data['asset_b'].index)
        cleaned_data = {
            'asset_a': cleaned_data['asset_a'].loc[common_index],
            'asset_b': cleaned_data['asset_b'].loc[common_index]
        }

        print(f" Cleaned data: {len(cleaned_data['asset_a'])} observations remaining")

        self.data = cleaned_data
        return self.data
    
    def construct_spread(self, method: str = 'cointegration',
                         lookback: int = 60) -> pd.DataFrame:
        """
        Constructing spread with the optimal hedge ratio
        
        For equity pairs, the spread is:
        X_t = log(P_A) - beta * log(P_B)

        Where beta is the hedge ratio 

        Params:
        -method: str
            -hedge ratio estimation method ('cointegration' or 'regression')
        -lookback: int
            -lookback period for rolling hedge ratio estiamation

        Returns:
        -pd.DataFrame
            -Spread data with columns: ['spread', 'asset_a_price', 'asset_b_price', 'hedge_ratio']
        """

        print(f"Constructing spread using {method} method")

        if not self.data:
            raise ValueError("No data available to construct the spread. Run load_from_csv() first.")
        
        asset_a_prices = self.data['asset_a']['Close']
        asset_b_prices = self.data['asset_b']['Close']

        #Estimate hedge ratio
        if method == 'cointegration':
            hedge_ratio = self._estimate_hedge_ratio_cointegration(
                asset_a_prices, asset_b_prices, lookback
            )
        elif method == 'regression':
            hedge_ratio = self._estimate_hedge_ratio_regression(
                asset_a_prices, asset_b_prices, lookback
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        #Construct the spread: X_t = log(P_A) - beta * log(P_B)
        log_a = np.log(asset_a_prices)
        log_b = np.log(asset_b_prices)

        spread = log_a - hedge_ratio * log_b

        #creating spread dataframe
        spread_df = pd.DataFrame({
            'spread': spread,
            'asset_a_price': asset_a_prices,
            'asset_b_price': asset_b_prices,
            'hedge_ratio': hedge_ratio,
            'log_a': log_a,
            'log_b': log_b
        })

        print(f"Spread constructed with the mean hedge ratio: {hedge_ratio.mean():.4f}")
        print(f"Spread range: [{spread.min():.4f}, {spread.max():.4f}]")
        print(f"Spread mean: {spread.mean():.4f}")
        print(f"Spread std: {spread.std():.4f}")

        return spread_df
    
    def _estimate_hedge_ratio_cointegration(self, asset_a: pd.Series,
                                            asset_b: pd.Series,
                                            lookback: int) -> pd.Series:
        """ 
        Estimate hedge ratio using rolling cointegration

        Params:
        -asset_a: pd.Series
            -Asset A prices (NVDA)
        -asset_b: pd.Series
            -Asset B prices (AMD)
        -lookback: int
            -rolling window size (in days)

        Returns:
        -pd.Series
            -Time-varying hedge ratio
        """

        hedge_ratios = pd.Series(index = asset_a.index, dtype = float)

        #use rolling window 
        for i in range(lookback, len(asset_a)):
            window_a = asset_a.iloc[i-lookback:i]
            window_b = asset_b.iloc[i-lookback:i]

            #OLS regression in log space
            beta = np.polyfit(np.log(window_b), np.log(window_a), 1)[0]
            hedge_ratios.iloc[i] = beta 

        #forward fill initial NaN values
        hedge_ratios = hedge_ratios.bfill()

        return hedge_ratios 
    
    def _estimate_hedge_ratio_regression(self, asset_a: pd.Series,
                                         asset_b: pd.Series,
                                         lookback: int) -> pd.Series:
        
        """
        Estimate the hedge ratio using rolling OLS regression

        Params:
        - asset_a: pd.Series
            -Asset A prices (NVDA)
        - asset_b: pd.Series
            - Asset B prices (AMD)
        - lookback: int
            - rolling window size

        Returns:
        -pd.Series
            -Time varying hedge ratio
        """

        hedge_ratios = pd.Series(index = asset_a.index, dtype = float)

        #Use rolling window
        for i in range(lookback, len(asset_a)):
            window_a = np.log(asset_a.iloc[i-lookback:i])
            window_b = np.log(asset_b.iloc[i-lookback:i])

            #OLS regression
            beta = np.polyfit(window_b, window_a, 1)[0]
            hedge_ratios.iloc[i] = beta

        #forward fill initial NaN values
        hedge_ratios = hedge_ratios.bfill()

        return hedge_ratios
    
    def calculate_spread_statistics(self, spread: pd.Series) -> Dict[str, float]:
        """
        Calculate key statistics for the spread

        Params:
        - spread: pd.Series
            -Spread time series

        Returns:
        - Dict[str, float]
            -Dictionary with mean, std, ADF statistic, half-life
        """

        from statsmodels.tsa.stattools import adfuller

        #Basic statistics
        mean = spread.mean()
        std = spread.std()

        #Stationarity test (ADF)
        adf_result = adfuller(spread.dropna())
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]

        #half life of mean reversion
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag 
        spread_lag_clean = spread_lag.dropna()
        spread_diff_clean = spread_diff.dropna()

        #align indicies
        common_idx = spread_lag_clean.index.intersection(spread_diff_clean.index)
        spread_lag_clean = spread_lag_clean.loc[common_idx]
        spread_diff_clean = spread_diff_clean.loc[common_idx]

        #regression for half-life
        beta = np.polyfit(spread_lag_clean, spread_diff_clean, 1)[0]
        half_life = -np.log(2) / beta if beta < 0 else np.inf

        stats = {
            'mean': mean,
            'std': std,
            'adf_statistic': adf_stat,
            'adf_pvalue': adf_pvalue, 
            'half_life': half_life,
            'is_stationary': adf_pvalue < 0.05
        }

        return stats 
    
if __name__ == "__main__":
    #testing the equity pairs pipeline 
    print("=" * 70)
    print("Equity Pairs Data Pipeline Test (NVDA/AMD)")
    print("="*70)

    #initialize pipelife
    pipeline = EquityPairsDataPipeline()
    current_dir = Path(__file__).resolve().parent

    #try to load CSV files
    try:
        #load data
        data = pipeline.load_from_csv(
            asset_a_path = str(current_dir / 'OHLCV_NVDA.csv'),
            asset_b_path = str(current_dir / 'OHLCV_AMD.csv')
        )

        #Clean data
        cleaned_data = pipeline.clean_data()

        #Construct spread
        spread_df = pipeline.construct_spread(method='cointegration', lookback = 60)

        #Calculate statistics
        stats = pipeline.calculate_spread_statistics(spread_df['spread'])

        print("\n" + "=" * 70)
        print("Spread Statistics (NVDA/AMD)")
        print("=" * 70)

        for key, value in stats.items():
            if isinstance(value, bool):
                print(f"{key}: {value}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")

        print(f"\n Spread data shape: {spread_df.shape}")
        print(f"Date range: {spread_df.index[0]} to {spread_df.index[-1]}")

    except FileNotFoundError:
        print("\n CSV not found")
        print("Please ensure OHLCV_NVDA.csv and OHLCV_AMD.csv are in correct path")
    except Exception as e:
        print(f"\n Error: {str(e)}")

    
