import os
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from khayyam import JalaliDatetime, JalaliDate


class TrafficDataProcessor:
    def __init__(self):

        self.config = {
            'dataset_url': "https://smoosavi.org/datasets/lstw/lstw.zip",
            'output_dir': "./processed_data",
            'temp_dir': "./temp_data",
            'api_token': 'ghazasnmtlhi06',
            'api_key': '4ffdd5b6bd6361871817d7d75f5ac8a1'
        }

        # Setup directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['temp_dir'], exist_ok=True)

        # Download and extract data
        self._download_and_extract()

        # Load data
        self.data = pd.read_csv(os.path.join(self.config['temp_dir'], 'lstw.csv'))
        print("Data loaded successfully")

    def _download_and_extract(self):

        url = self.config['dataset_url']
        res = requests.get(url)

        with zipfile.ZipFile(BytesIO(res.content)) as zf:
            zf.extractall(self.config['temp_dir'])

    def clean_data(self):

        print("\nStarting comprehensive data analysis...")

        # 1. Convert to DataFrame and show basic info
        df = pd.DataFrame(self.data)
        print("\nBasic DataFrame Info:")
        print(f"- Shape: {df.shape}")
        print(f"- Size: {df.size}")
        print(f"- Dimensions: {df.ndim}")

        # 2. Display data types
        print("\nData Types:")
        print(df.dtypes)

        # 3. Show detailed info
        print("\nDetailed Info:")
        df.info()

        # 3. Encode categorical variables (Day/Night as 0/1)
        if 'Day/Night' in self.data.columns:
            self.data['Day/Night'] = self.data['Day/Night'].map({'Day': 0, 'Night': 1})

        # 5. Show missing values
        print("\nMissing Values Summary:")
        print(df.isna().sum())

        # 6. Detect outliers using IQR method
        print("\nOutlier Detection (IQR Method):")
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].nunique() > 5:  # Only check columns with sufficient variation
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                print(f"{col}: {len(outliers)} outliers detected (bounds: {lower_bound:.2f} to {upper_bound:.2f})")

        # 7. Convert datetime to Jalali (if datetime column exists)
        print("\nDate Conversion:")
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
        for col in datetime_cols:
            try:
                df[f'{col}_Jalali'] = df[col].apply(
                    lambda x: JalaliDatetime(x).strftime('%Y-%m-%d') if not pd.isna(x) else None)
                print(f"Converted {col} to Jalali calendar")
            except Exception as e:
                print(f"Couldn't convert {col}: {str(e)}")

        # Update the main dataframe
        self.data = df

        print("\nAnalysis completed without removing any columns")
        return self

    def optimize_size(self):

        # 1. Create a working copy
        df = self.data.copy()

        # 2. Remove unique identifier columns (like IDs)
        # These columns don't provide analytical value and waste space
        unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
        df.drop(columns=unique_cols, inplace=True, errors='ignore')
        print(f"Removed {len(unique_cols)} unique identifier columns")

        # 3. Optimize numeric data types
        for col in df.select_dtypes(include=['int']):
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float']):
            df[col] = pd.to_numeric(df[col], downcast='float')

        # 4. Convert objects to category where appropriate
        for col in df.select_dtypes(include=['object']):
            if df[col].nunique() / len(df) < 0.5:  # If <50% unique values
                df[col] = df[col].astype('category')

        # 5. Advanced Parquet compression
        output_path = os.path.join(self.config['output_dir'], 'optimized_traffic.parquet')

        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='gzip',  # Good balance of size
            compression_level=11,
            row_group_size=100000  # Better compression for large files
        )

        # 6. Verify results
        final_size = os.path.getsize(output_path) / (1024 ** 2)
        print(f"\nOptimization Results:")
        print(f"- Original size: {self.data.memory_usage(deep=True).sum() / (1024 ** 3):.2f}GB")
        print(f"- Final size: {final_size:.2f}MB")
        print(f"- Columns removed: {len(unique_cols)}")

        if final_size > 600:
            print("Error!!!")

    def split_and_save(self):

        X, y = self.data.drop('Severity', axis=1), self.data['Severity']

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Save processed data
        self.data.to_csv('traffic_data_cleaned.csv', index=False)

        return X_train, X_test, y_train, y_test

    # Usage example
if __name__ == "__main__":
    print("Starting traffic data processing...")
    processor = TrafficDataProcessor()
    processor.clean_data()
    X_train, X_test, y_train, y_test = processor.split_and_save()
    print("\nProcessing complete!")