import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import ta
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data has required columns"""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data and calculate technical indicators"""
        # Validate input data
        self.validate_data(df)
        
        # Create copy to avoid modifying original data
        processed_df = df.copy()
        
        # Calculate technical indicators
        # RSI
        processed_df['RSI'] = ta.momentum.RSIIndicator(
            processed_df['Close'], window=14
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(processed_df['Close'])
        processed_df['MACD'] = macd.macd()
        processed_df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(processed_df['Close'])
        processed_df['BB_upper'] = bollinger.bollinger_hband()
        processed_df['BB_lower'] = bollinger.bollinger_lband()
        processed_df['BB_middle'] = bollinger.bollinger_mavg()
        
        # Moving Averages
        processed_df['EMA_9'] = ta.trend.EMAIndicator(
            processed_df['Close'], window=9
        ).ema_indicator()
        processed_df['SMA_30'] = ta.trend.SMAIndicator(
            processed_df['Close'], window=30
        ).sma_indicator()
        
        # ADX (Average Directional Index)
        processed_df['ADX'] = ta.trend.ADXIndicator(
            processed_df['High'], processed_df['Low'], processed_df['Close']
        ).adx()
        
        # OBV (On Balance Volume)
        processed_df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            processed_df['Close'], processed_df['Volume']
        ).on_balance_volume()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            processed_df['High'], 
            processed_df['Low'], 
            processed_df['Close']
        )
        processed_df['Stoch_K'] = stoch.stoch()
        processed_df['Stoch_D'] = stoch.stoch_signal()
        
        # Average True Range (ATR)
        processed_df['ATR'] = ta.volatility.AverageTrueRange(
            processed_df['High'],
            processed_df['Low'],
            processed_df['Close']
        ).average_true_range()
        
        # Fibonacci Retracement Levels
        processed_df['Fib_38.2'] = self._calculate_fibonacci_level(processed_df, 0.382)
        processed_df['Fib_50.0'] = self._calculate_fibonacci_level(processed_df, 0.5)
        processed_df['Fib_61.8'] = self._calculate_fibonacci_level(processed_df, 0.618)
        
        # Handle missing values
        processed_df.fillna(method='ffill', inplace=True)
        processed_df.fillna(method='bfill', inplace=True)
        
        return processed_df
    
    def generate_technical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical analysis summary"""
        processed_df = self.process(df)
        last_row = processed_df.iloc[-1]
        
        # Determine trends
        price_trend = self._determine_trend(processed_df['Close'])
        volume_trend = self._determine_trend(processed_df['Volume'])
        
        # Calculate support and resistance
        support, resistance = self._calculate_support_resistance(processed_df)
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'trends': {
                'price': price_trend,
                'volume': volume_trend
            },
            'indicators': {
                'rsi': {
                    'value': round(last_row['RSI'], 2),
                    'signal': self._interpret_rsi(last_row['RSI'])
                },
                'macd': {
                    'value': round(last_row['MACD'], 2),
                    'signal': round(last_row['MACD_Signal'], 2),
                    'interpretation': self._interpret_macd(
                        last_row['MACD'], 
                        last_row['MACD_Signal']
                    )
                },
                'bollinger_bands': {
                    'upper': round(last_row['BB_upper'], 2),
                    'middle': round(last_row['BB_middle'], 2),
                    'lower': round(last_row['BB_lower'], 2),
                    'position': self._interpret_bollinger_position(last_row)
                },
                'stochastic': {
                    'k': round(last_row['Stoch_K'], 2),
                    'd': round(last_row['Stoch_D'], 2),
                    'signal': self._interpret_stochastic(
                        last_row['Stoch_K'],
                        last_row['Stoch_D']
                    )
                }
            },
            'support_resistance': {
                'support_levels': [round(level, 2) for level in support],
                'resistance_levels': [round(level, 2) for level in resistance]
            },
            'fibonacci_levels': {
                '38.2%': round(last_row['Fib_38.2'], 2),
                '50.0%': round(last_row['Fib_50.0'], 2),
                '61.8%': round(last_row['Fib_61.8'], 2)
            },
            'volatility': {
                'atr': round(last_row['ATR'], 2),
                'bollinger_bandwidth': self._calculate_bollinger_bandwidth(last_row)
            }
        }
        
        return analysis
    
    def _determine_trend(self, series: pd.Series, window: int = 20) -> str:
        """Determine trend direction using moving averages"""
        ma = series.rolling(window=window).mean()
        current_price = series.iloc[-1]
        ma_current = ma.iloc[-1]
        ma_prev = ma.iloc[-2]
        
        if current_price > ma_current and ma_current > ma_prev:
            return "Uptrend"
        elif current_price < ma_current and ma_current < ma_prev:
            return "Downtrend"
        else:
            return "Sideways"
    
    def _calculate_support_resistance(
        self, 
        df: pd.DataFrame, 
        window: int = 20,
        num_levels: int = 3
    ) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels using local extrema"""
        prices = df['Close'].values
        
        # Find local minima and maxima
        local_min_idx = argrelextrema(prices, np.less, order=window)[0]
        local_max_idx = argrelextrema(prices, np.greater, order=window)[0]
        
        support_levels = sorted(prices[local_min_idx], reverse=True)[:num_levels]
        resistance_levels = sorted(prices[local_max_idx], reverse=True)[:num_levels]
        
        return support_levels, resistance_levels
    
    def _calculate_fibonacci_level(
        self, 
        df: pd.DataFrame, 
        level: float
    ) -> pd.Series:
        """Calculate Fibonacci retracement level"""
        high = df['High'].rolling(window=20).max()
        low = df['Low'].rolling(window=20).min()
        return high - (high - low) * level
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI values"""
        if rsi >= 70:
            return "Overbought"
        elif rsi <= 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def _interpret_macd(self, macd: float, signal: float) -> str:
        """Interpret MACD signals"""
        if macd > signal:
            return "Bullish"
        elif macd < signal:
            return "Bearish"
        else:
            return "Neutral"
    
    def _interpret_bollinger_position(self, row: pd.Series) -> str:
        """Interpret price position relative to Bollinger Bands"""
        price = row['Close']
        if price > row['BB_upper']:
            return "Above Upper Band"
        elif price < row['BB_lower']:
            return "Below Lower Band"
        else:
            return "Within Bands"
    
    def _interpret_stochastic(self, k: float, d: float) -> str:
        """Interpret Stochastic Oscillator signals"""
        if k > 80 and d > 80:
            return "Overbought"
        elif k < 20 and d < 20:
            return "Oversold"
        elif k > d:
            return "Bullish"
        else:
            return "Bearish"
    
    def _calculate_bollinger_bandwidth(self, row: pd.Series) -> float:
        """Calculate Bollinger Bandwidth for volatility measurement"""
        return ((row['BB_upper'] - row['BB_lower']) / row['BB_middle']) * 100
    
    def generate_market_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive market summary"""
        analysis = self.generate_technical_analysis(df)
        last_price = df['Close'].iloc[-1]
        
        summary = {
            'current_price': round(last_price, 2),
            'price_change_1d': self._calculate_price_change(df, days=1),
            'price_change_5d': self._calculate_price_change(df, days=5),
            'volume_analysis': self._analyze_volume(df),
            'technical_analysis': analysis,
            'risk_metrics': self._calculate_risk_metrics(df),
            'momentum_signals': self._generate_momentum_signals(df)
        }
        
        return summary
    
    def _calculate_price_change(
        self, 
        df: pd.DataFrame, 
        days: int
    ) -> Dict[str, float]:
        """Calculate price change over specified period"""
        if len(df) < days:
            return {'absolute': 0, 'percentage': 0}
            
        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-days-1]
        
        absolute_change = round(current_price - previous_price, 2)
        percentage_change = round((absolute_change / previous_price) * 100, 2)
        
        return {
            'absolute': absolute_change,
            'percentage': percentage_change
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        return {
            'current_volume': int(current_volume),
            'average_volume': int(avg_volume),
            'volume_ratio': round(current_volume / avg_volume, 2),
            'trend': self._determine_trend(df['Volume'])
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various risk metrics"""
        returns = df['Close'].pct_change()
        
        return {
            'volatility': round(returns.std() * np.sqrt(252) * 100, 2),  # Annualized
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(df['Close']),
            'beta': self._calculate_beta(returns)  # Assuming market returns available
        }
    
    def _calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.01
    ) -> float:
        """Calculate Sharpe Ratio"""
        excess_returns = returns - risk_free_rate/252
        return round(np.sqrt(252) * excess_returns.mean() / returns.std(), 2)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return round(drawdown.min() * 100, 2)
    
    def _calculate_beta(
        self, 
        returns: pd.Series, 
        market_returns: pd.Series = None
    ) -> float:
        """Calculate Beta (market sensitivity)"""
        if market_returns is None:
            return 1.0  # Default to market beta if no comparison available
        
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return round(covariance / market_variance, 2)
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate momentum-based trading signals"""
        processed_df = self.process(df)
        last_row = processed_df.iloc[-1]
        
        signals = {
            'overall_trend': self._determine_trend(df['Close']),
            'rsi_signal': self._interpret_rsi(last_row['RSI']),
            'macd_signal': self._interpret_macd(
                last_row['MACD'],
                last_row['MACD_Signal']
            ),
            'stochastic_signal': self._interpret_stochastic(
                last_row['Stoch_K'],
                last_row['Stoch_D']
            ),
            'bollinger_signal': self._interpret_bollinger_position(last_row)
        }
        
        # Generate composite signal
        signals['composite_signal'] = self._generate_composite_signal(signals)
        
        return signals

    def _generate_composite_signal(self, signals: Dict[str, str]) -> str:
        """Generate a composite trading signal based on multiple indicators"""
        bullish_count = 0
        bearish_count = 0

        # Count bullish and bearish signals
        if signals['overall_trend'] == "Uptrend":
            bullish_count += 1
        elif signals['overall_trend'] == "Downtrend":
            bearish_count += 1

        if signals['rsi_signal'] == "Oversold":
            bullish_count += 1
        elif signals['rsi_signal'] == "Overbought":
            bearish_count += 1

        if signals['macd_signal'] == "Bullish":
            bullish_count += 1
        elif signals['macd_signal'] == "Bearish":
            bearish_count += 1

        if signals['stochastic_signal'] == "Oversold":
            bullish_count += 1
        elif signals['stochastic_signal'] == "Overbought":
            bearish_count += 1

        # Generate final signal
        if bullish_count >= 3:
            return "Strong Buy"
        elif bullish_count >= 2:
            return "Buy"
        elif bearish_count >= 3:
            return "Strong Sell"
        elif bearish_count >= 2:
            return "Sell"
        else:
            return "Hold"

    def prepare_training_data(
            self,
            df: pd.DataFrame,
            sequence_length: int = 60,
            prediction_horizon: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training with sequential data

        Args:
            df: DataFrame with processed indicators
            sequence_length: Number of time steps to include in each sequence
            prediction_horizon: Number of days ahead to predict

        Returns:
            X: Training sequences
            y: Target values
        """
        processed_df = self.process(df)

        # Select features for training
        feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
            'BB_upper', 'BB_lower', 'EMA_9', 'SMA_30', 'ADX',
            'OBV', 'Stoch_K', 'Stoch_D', 'ATR'
        ]

        # Normalize features
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(processed_df[feature_columns])

        X = []
        y = []

        # Create sequences
        for i in range(sequence_length, len(normalized_data) - prediction_horizon):
            X.append(normalized_data[i - sequence_length:i])
            y.append(processed_df['Close'].iloc[i + prediction_horizon])

        return np.array(X), np.array(y)

    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced trading and analysis metrics"""
        processed_df = self.process(df)

        metrics = {
            'volatility_metrics': self._calculate_volatility_metrics(processed_df),
            'momentum_metrics': self._calculate_momentum_metrics(processed_df),
            'volume_metrics': self._calculate_volume_metrics(processed_df),
            'trend_metrics': self._calculate_trend_metrics(processed_df),
            'pattern_recognition': self._identify_patterns(processed_df)
        }

        return metrics

    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various volatility metrics"""
        returns = df['Close'].pct_change()

        metrics = {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'parkinson_volatility': self._calculate_parkinson_volatility(df),
            'garman_klass_volatility': self._calculate_garman_klass_volatility(df)
        }

        return {k: round(v, 4) for k, v in metrics.items()}

    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Parkinson volatility estimator"""
        high_low_ratio = np.log(df['High'] / df['Low'])
        return np.sqrt(1 / (4 * np.log(2)) * high_low_ratio.pow(2).mean())

    def _calculate_garman_klass_volatility(self, df: pd.DataFrame) -> float:
        """Calculate Garman-Klass volatility estimator"""
        high_low = 0.5 * np.log(df['High'] / df['Low']).pow(2)
        close_open = 2 * np.log(df['Close'] / df['Open']).pow(2)
        return np.sqrt(high_low.mean() - (2 * np.log(2) - 1) * close_open.mean())

    def _calculate_momentum_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based metrics"""
        returns = df['Close'].pct_change()

        metrics = {
            'momentum_1d': returns.iloc[-1],
            'momentum_5d': df['Close'].pct_change(5).iloc[-1],
            'momentum_20d': df['Close'].pct_change(20).iloc[-1],
            'rate_of_change': (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1),
            'acceleration': returns.diff().mean()
        }

        return {k: round(v, 4) for k, v in metrics.items()}

    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based metrics"""
        metrics = {
            'volume_ma_ratio': df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1],
            'volume_trend': self._determine_trend(df['Volume']),
            'volume_volatility': df['Volume'].pct_change().std(),
            'price_volume_correlation': df['Close'].pct_change().corr(df['Volume'].pct_change())
        }

        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    def _calculate_trend_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-related metrics"""
        metrics = {
            'adx_trend_strength': df['ADX'].iloc[-1],
            'price_ma_ratio': df['Close'].iloc[-1] / df['SMA_30'].iloc[-1],
            'trend_direction': self._determine_trend(df['Close']),
            'trend_consistency': self._calculate_trend_consistency(df)
        }

        return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate trend consistency metric"""
        returns = df['Close'].pct_change()
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        return positive_days / total_days

    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Identify common chart patterns"""
        patterns = {
            'double_top': self._check_double_top(df),
            'double_bottom': self._check_double_bottom(df),
            'head_and_shoulders': self._check_head_and_shoulders(df),
            'triangle': self._check_triangle_pattern(df)
        }

        return patterns

    def _check_double_top(self, df: pd.DataFrame, threshold: float = 0.02) -> bool:
        """Check for double top pattern"""
        highs = df['High'].rolling(window=5).max()
        peaks = argrelextrema(highs.values, np.greater_equal, order=5)[0]

        if len(peaks) < 2:
            return False

        last_two_peaks = peaks[-2:]
        peak_values = highs.iloc[last_two_peaks]

        return abs(peak_values.iloc[0] - peak_values.iloc[1]) / peak_values.iloc[0] < threshold

    def _check_double_bottom(self, df: pd.DataFrame, threshold: float = 0.02) -> bool:
        """Check for double bottom pattern"""
        lows = df['Low'].rolling(window=5).min()
        troughs = argrelextrema(lows.values, np.less_equal, order=5)[0]

        if len(troughs) < 2:
            return False

        last_two_troughs = troughs[-2:]
        trough_values = lows.iloc[last_two_troughs]

        return abs(trough_values.iloc[0] - trough_values.iloc[1]) / trough_values.iloc[0] < threshold

    def _check_head_and_shoulders(self, df: pd.DataFrame, threshold: float = 0.02) -> bool:
        """Check for head and shoulders pattern"""
        highs = df['High'].rolling(window=5).max()
        peaks = argrelextrema(highs.values, np.greater_equal, order=5)[0]

        if len(peaks) < 3:
            return False

        last_three_peaks = peaks[-3:]
        peak_values = highs.iloc[last_three_peaks]

        middle_higher = (peak_values.iloc[1] > peak_values.iloc[0] and
                         peak_values.iloc[1] > peak_values.iloc[2])
        shoulders_similar = (abs(peak_values.iloc[0] - peak_values.iloc[2]) /
                             peak_values.iloc[0] < threshold)

        return middle_higher and shoulders_similar

    def _check_triangle_pattern(self, df: pd.DataFrame, window: int = 20) -> bool:
        """Check for triangle pattern (convergence of highs and lows)"""
        recent_data = df.tail(window)

        high_slope = np.polyfit(range(window), recent_data['High'], 1)[0]
        low_slope = np.polyfit(range(window), recent_data['Low'], 1)[0]

        # Check if highs are decreasing and lows are increasing
        return high_slope < 0 and low_slope > 0