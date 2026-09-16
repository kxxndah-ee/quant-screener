import unittest
import pandas as pd
from src.core.scoring import evaluate_intraday_outlook, evaluate_stock_latest


class TestOutlook(unittest.TestCase):
    def test_breakout_outlook(self):
        # High open gain, high candle support
        row = pd.Series({
            'Open': 10000.0,
            'High': 10600.0,
            'Low': 9950.0,
            'Close': 10500.0,
            'VolRatio': 1.5,
            'Volume': 150000
        })
        res = evaluate_intraday_outlook(row, prev_close=10000.0)
        self.assertIn('수급돌파', res['tag'])
        self.assertGreater(res['open_gain_pct'], 2.0)

    def test_upper_shadow_warning(self):
        # Spiked to 11000 but dumped to 10100 (upper shadow 900)
        row = pd.Series({
            'Open': 10000.0,
            'High': 11000.0,
            'Low': 9900.0,
            'Close': 10100.0,
            'VolRatio': 1.2,
            'Volume': 100000
        })
        res = evaluate_intraday_outlook(row, prev_close=10000.0)
        self.assertIn('윗꼬리', res['tag'])

    def test_pullback_support(self):
        # Around open, holding low
        row = pd.Series({
            'Open': 10000.0,
            'High': 10150.0,
            'Low': 9950.0,
            'Close': 10050.0,
            'VolRatio': 0.7,
            'Volume': 50000
        })
        res = evaluate_intraday_outlook(row, prev_close=10000.0)
        self.assertIn('눌림지지', res['tag'])

    def test_below_open_bearish(self):
        # Drop below open
        row = pd.Series({
            'Open': 10000.0,
            'High': 10050.0,
            'Low': 9700.0,
            'Close': 9750.0,
            'VolRatio': 0.9,
            'Volume': 80000
        })
        res = evaluate_intraday_outlook(row, prev_close=10000.0)
        self.assertIn('시초하회', res['tag'])


if __name__ == '__main__':
    unittest.main()
