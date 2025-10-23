#!/usr/bin/env python3
"""
Test TwelveData API to understand the correct symbol format
"""

import requests
import json

def test_api_symbols():
    api_key = "0e24ff3eb6ef415dba0cebcf04593e4f"
    base_url = "https://api.twelvedata.com/time_series"

    # Test different symbol formats
    test_symbols = [
        "EUR/USD",  # Standard format with slash
        "EURUSD",   # No slash
        "EUR-USD",  # With dash
        "USD/EUR",  # Reversed
        "GBP/USD",
        "GBPUSD"
    ]

    for symbol in test_symbols:
        params = {
            'symbol': symbol,
            'interval': '1d',
            'outputsize': '5',  # Just test with 5 records
            'apikey': api_key
        }

        print(f"\nTesting symbol: {symbol}")
        try:
            response = requests.get(base_url, params=params)
            print(f"Status code: {response.status_code}")

            data = response.json()
            print(f"Response keys: {list(data.keys())}")

            if 'values' in data:
                print(f"Data found! Records: {len(data['values'])}")
                if data['values']:
                    print(f"Sample record: {data['values'][0]}")
            elif 'message' in data:
                print(f"Message: {data['message']}")
            else:
                print(f"Full response: {json.dumps(data, indent=2)}")

        except Exception as e:
            print(f"Error: {str(e)}")

def test_forex_list_endpoint():
    """Test if there's a forex list endpoint"""
    api_key = "0e24ff3eb6ef415dba0cebcf04593e4f"

    # Try different endpoints to find available forex pairs
    endpoints = [
        "https://api.twelvedata.com/forex_pairs",
        "https://api.twelvedata.com/symbol_search?symbol=EUR&instrument=forex",
        "https://api.twelvedata.com/symbol_search?symbol=USD"
    ]

    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        try:
            if "symbol_search" in endpoint:
                response = requests.get(f"{endpoint}&apikey={api_key}")
            else:
                response = requests.get(f"{endpoint}?apikey={api_key}")

            print(f"Status: {response.status_code}")
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)[:500]}...")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Testing TwelveData API symbol formats...")
    test_api_symbols()

    print("\n" + "="*60)
    print("Testing forex list endpoints...")
    test_forex_list_endpoint()