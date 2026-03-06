import os
import pandas as pd
import time


def traffic_stream(folder_path, delay=0.5):
    """
    Simulates real-time traffic from CIC-IDS2017 folder
    """
    csv_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".parquet")
    ])

    for file in csv_files:
        df = pd.read_parquet(file)
        df = df.select_dtypes(include=['number'])
        df.replace([float('inf'), -float('inf')], 0, inplace=True)
        df.fillna(0, inplace=True)

        for i in range(len(df)):
            yield df.iloc[[i]]
            time.sleep(delay)

