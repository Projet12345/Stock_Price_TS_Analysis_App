#!/usr/bin/env python3
"""
Quick AI Revolution Dataset Generator
Author: Anderson Nguetoum
"""

import csv
import random
from datetime import datetime, timedelta

# Simplified AI companies
AI_COMPANIES = {
    'NVDA': {'name': 'NVIDIA', 'base_price': 200},
    'MSFT': {'name': 'Microsoft', 'base_price': 300},
    'GOOGL': {'name': 'Alphabet', 'base_price': 2500},
    'AMZN': {'name': 'Amazon', 'base_price': 3000},
    'TSLA': {'name': 'Tesla', 'base_price': 200}
}

def generate_quick_dataset():
    """Generate quick AI dataset"""
    print("🤖 Generating AI Revolution Stock Dataset...")

    # Generate 2 years of data (500 trading days)
    data = []
    base_date = datetime(2022, 1, 1)

    for symbol, info in AI_COMPANIES.items():
        print(f"📈 Generating {symbol}...")
        price = info['base_price']

        for i in range(500):
            date = base_date + timedelta(days=i*1.4)  # Skip weekends roughly

            # Random price movement
            change = random.uniform(-0.05, 0.05)
            price = price * (1 + change)

            # Generate OHLC
            open_price = price * random.uniform(0.98, 1.02)
            high_price = price * random.uniform(1.0, 1.05)
            low_price = price * random.uniform(0.95, 1.0)
            volume = random.randint(1000000, 50000000)

            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Company': info['name'],
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(price, 2),
                'Volume': volume,
                'AI_Sentiment': random.randint(30, 90)
            })

    return data

def save_csv(data):
    """Save to CSV"""
    filename = 'data/ai_revolution_stock_data.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Dataset saved: {filename}")
    print(f"📊 Total records: {len(data)}")
    return filename

if __name__ == "__main__":
    data = generate_quick_dataset()
    save_csv(data)
    print("🚀 AI Revolution dataset ready!")