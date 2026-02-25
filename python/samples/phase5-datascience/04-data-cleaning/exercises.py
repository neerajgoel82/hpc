"""
Data Cleaning Exercises
========================
Practice problems covering all data cleaning concepts from this module.

Topics covered:
- Missing data handling
- Data types and conversion
- Outlier detection
- Feature engineering
- Scaling and normalization
- Text preprocessing

Run: python exercises.py
"""

import numpy as np
import pandas as pd
import re
import string

def exercise_1():
    """Missing data handling - Multiple strategies"""
    print("\nExercise 1: Missing Data Strategies")
    print("-" * 40)

    # Create dataset with various missing patterns
    np.random.seed(42)
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
        'Age': [25, np.nan, 35, 30, np.nan, 28, 32, np.nan],
        'Salary': [50000, 60000, np.nan, 55000, 70000, np.nan, 65000, 58000],
        'Department': ['IT', 'HR', 'IT', np.nan, 'HR', 'Finance', np.nan, 'IT'],
        'YearsExp': [2, 5, np.nan, 4, 8, 3, 6, np.nan]
    })

    print("Dataset with missing values:")
    print(df)

    print("\nMissing value counts:")
    print(df.isnull().sum())

    print("\nMissing percentages:")
    print((df.isnull().sum() / len(df) * 100).round(2))

    # Strategy 1: Fill numeric with mean
    df_filled = df.copy()
    df_filled['Age'].fillna(df_filled['Age'].mean(), inplace=True)
    df_filled['Salary'].fillna(df_filled['Salary'].median(), inplace=True)
    df_filled['YearsExp'].fillna(df_filled['YearsExp'].mean(), inplace=True)

    print("\nNumeric columns filled with mean/median:")
    print(df_filled[['Age', 'Salary', 'YearsExp']])

    # Strategy 2: Fill categorical with mode
    df_filled['Department'].fillna(df_filled['Department'].mode()[0], inplace=True)

    print("\nComplete filled dataset:")
    print(df_filled)

    # Strategy 3: Group-based imputation
    df_grouped = df.copy()
    df_grouped['Salary_GroupFilled'] = df_grouped.groupby('Department')['Salary'].transform(
        lambda x: x.fillna(x.mean())
    )

    print("\nGroup-based salary imputation:")
    print(df_grouped[['Department', 'Salary', 'Salary_GroupFilled']])

def exercise_2():
    """Data type conversion and optimization"""
    print("\nExercise 2: Data Type Optimization")
    print("-" * 40)

    # Create dataset with inefficient types
    df = pd.DataFrame({
        'ID': range(1, 1001),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'Score': np.random.randint(0, 100, 1000),
        'Active': np.random.choice(['Yes', 'No'], 1000),
        'Date': pd.date_range('2024-01-01', periods=1000, freq='h')
    })

    print("Original DataFrame info:")
    print(df.dtypes)
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum():,} bytes")

    # Optimize data types
    df_optimized = df.copy()
    df_optimized['ID'] = df_optimized['ID'].astype('int16')
    df_optimized['Category'] = df_optimized['Category'].astype('category')
    df_optimized['Score'] = df_optimized['Score'].astype('int8')
    df_optimized['Active'] = df_optimized['Active'].map({'Yes': True, 'No': False})

    print("\nOptimized DataFrame info:")
    print(df_optimized.dtypes)
    print(f"\nMemory usage: {df_optimized.memory_usage(deep=True).sum():,} bytes")

    original_size = df.memory_usage(deep=True).sum()
    optimized_size = df_optimized.memory_usage(deep=True).sum()
    savings = (1 - optimized_size / original_size) * 100

    print(f"\nMemory savings: {savings:.1f}%")

def exercise_3():
    """Outlier detection and handling"""
    print("\nExercise 3: Outlier Detection and Handling")
    print("-" * 40)

    # Create dataset with outliers
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 95)
    outliers = [200, 205, 10, 15, 195]
    data = np.concatenate([normal_data, outliers])

    df = pd.DataFrame({'Value': data})

    print("Dataset statistics:")
    print(df['Value'].describe())

    # Method 1: Z-score
    df['Z_Score'] = (df['Value'] - df['Value'].mean()) / df['Value'].std()
    outliers_z = df[np.abs(df['Z_Score']) > 3]

    print(f"\nOutliers by Z-score (|Z| > 3): {len(outliers_z)}")
    print(outliers_z[['Value', 'Z_Score']].head())

    # Method 2: IQR
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]

    print(f"\nOutliers by IQR method: {len(outliers_iqr)}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(outliers_iqr['Value'].values)

    # Method 3: Capping
    df['Value_Capped'] = df['Value'].clip(lower=lower_bound, upper=upper_bound)

    print("\nBefore and after capping (extremes):")
    extreme_idx = df['Value'].nlargest(3).index.tolist() + df['Value'].nsmallest(3).index.tolist()
    print(df.loc[extreme_idx, ['Value', 'Value_Capped']].sort_values('Value'))

def exercise_4():
    """Feature engineering - Multiple techniques"""
    print("\nExercise 4: Feature Engineering")
    print("-" * 40)

    # Create e-commerce transaction data
    np.random.seed(42)
    df = pd.DataFrame({
        'TransactionID': range(1, 11),
        'Date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'Price': [100, 150, 200, 120, 180, 90, 160, 140, 110, 190],
        'Quantity': [2, 1, 3, 2, 1, 4, 2, 3, 2, 1],
        'Discount': [10, 0, 20, 15, 10, 5, 0, 10, 5, 15]
    })

    print("Original data:")
    print(df)

    # Feature 1: Total revenue
    df['Revenue'] = df['Price'] * df['Quantity']

    # Feature 2: Discount percentage
    df['Discount_Pct'] = (df['Discount'] / df['Price'] * 100).round(2)

    # Feature 3: Final price
    df['Final_Price'] = df['Price'] - df['Discount']

    # Feature 4: DateTime features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayName'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Feature 5: Price category
    df['Price_Category'] = pd.cut(df['Price'],
                                   bins=[0, 100, 150, 300],
                                   labels=['Low', 'Medium', 'High'])

    # Feature 6: High discount flag
    df['High_Discount'] = (df['Discount_Pct'] > 10).astype(int)

    print("\nWith engineered features:")
    print(df[['TransactionID', 'Revenue', 'Discount_Pct', 'Is_Weekend',
              'Price_Category', 'High_Discount']])

    print("\nAggregate features:")
    print(f"Average revenue: ${df['Revenue'].mean():.2f}")
    print(f"Total revenue: ${df['Revenue'].sum():.2f}")
    print(f"Weekend transactions: {df['Is_Weekend'].sum()}")
    print(f"High discount transactions: {df['High_Discount'].sum()}")

def exercise_5():
    """Scaling comparison"""
    print("\nExercise 5: Scaling Methods Comparison")
    print("-" * 40)

    # Create features with different scales
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(20, 70, 10),
        'Salary': np.random.randint(30000, 120000, 10),
        'Years_Experience': np.random.randint(0, 30, 10)
    })

    print("Original data:")
    print(df)

    print("\nOriginal statistics:")
    print(df.describe().round(2))

    # StandardScaler
    df_std = df.copy()
    for col in df_std.columns:
        mean = df_std[col].mean()
        std = df_std[col].std()
        df_std[col] = (df_std[col] - mean) / std

    print("\nStandardized (mean=0, std=1):")
    print(df_std.round(3))

    # MinMaxScaler
    df_minmax = df.copy()
    for col in df_minmax.columns:
        min_val = df_minmax[col].min()
        max_val = df_minmax[col].max()
        df_minmax[col] = (df_minmax[col] - min_val) / (max_val - min_val)

    print("\nMin-Max scaled (0 to 1):")
    print(df_minmax.round(3))

    # RobustScaler
    df_robust = df.copy()
    for col in df_robust.columns:
        median = df_robust[col].median()
        Q1 = df_robust[col].quantile(0.25)
        Q3 = df_robust[col].quantile(0.75)
        IQR = Q3 - Q1
        df_robust[col] = (df_robust[col] - median) / IQR

    print("\nRobust scaled (median, IQR):")
    print(df_robust.round(3))

def exercise_6():
    """Text preprocessing pipeline"""
    print("\nExercise 6: Text Preprocessing Pipeline")
    print("-" * 40)

    # Sample product reviews
    reviews = [
        "AMAZING product!!! I love it so much. Best purchase ever!",
        "Terrible quality :( Do NOT buy this. Waste of money.",
        "Pretty good for the price. Would recommend to friends.",
        "Not bad, but I've seen better. It's okay I guess...",
        "Excellent! Works perfectly! 5 stars!!! #recommended"
    ]

    df = pd.DataFrame({'Review': reviews})

    print("Original reviews:")
    for i, review in enumerate(reviews, 1):
        print(f"{i}. {review}")

    def preprocess_text(text):
        # Lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    # Simple stopwords
    stopwords = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                 'from', 'has', 'he', 'i', 'in', 'is', 'it', 'of', 'on',
                 'that', 'the', 'to', 'was', 'will', 'with', 'this'}

    # Apply preprocessing
    df['Cleaned'] = df['Review'].apply(preprocess_text)

    # Remove stopwords
    df['No_Stopwords'] = df['Cleaned'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stopwords])
    )

    # Count tokens
    df['Token_Count'] = df['No_Stopwords'].apply(lambda x: len(x.split()))

    print("\nPreprocessed reviews:")
    print(df[['Review', 'No_Stopwords', 'Token_Count']])

def exercise_7():
    """Complete data cleaning pipeline"""
    print("\nExercise 7: Complete Data Cleaning Pipeline")
    print("-" * 40)

    # Create messy dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
        'name': ['Alice', 'Bob', np.nan, 'David', 'Eve', 'Frank'],
        'age': [25, 350, 35, 30, np.nan, 28],  # 350 is an outlier
        'income': ['$50,000', '$60,000', '$75,000', np.nan, '$70,000', '$65,000'],
        'purchase_date': ['2024-01-15', '2024/02/20', '2024-03-25', '2024-04-10', '2024-05-15', '2024-06-20'],
        'category': ['gold', 'SILVER', 'Gold', 'bronze', 'Gold', np.nan]
    })

    print("Original messy data:")
    print(df)

    print("\nIssues identified:")
    print(f"  1. Missing values: {df.isnull().sum().sum()}")
    print(f"  2. Income has $ and commas")
    print(f"  3. Date formats are inconsistent")
    print(f"  4. Category has case inconsistencies")
    print(f"  5. Age has outlier (350)")

    # Clean the data
    df_clean = df.copy()

    # 1. Handle missing name
    df_clean['name'].fillna('Unknown', inplace=True)

    # 2. Handle outlier in age
    age_median = df_clean['age'].median()
    df_clean.loc[df_clean['age'] > 100, 'age'] = age_median

    # 3. Fill missing age with median
    df_clean['age'].fillna(df_clean['age'].median(), inplace=True)

    # 4. Clean income column
    df_clean['income'] = df_clean['income'].str.replace('$', '').str.replace(',', '')
    df_clean['income'] = pd.to_numeric(df_clean['income'], errors='coerce')
    df_clean['income'].fillna(df_clean['income'].median(), inplace=True)
    df_clean['income'] = df_clean['income'].astype(int)

    # 5. Standardize date format
    df_clean['purchase_date'] = pd.to_datetime(df_clean['purchase_date'], format='mixed')

    # 6. Standardize category (lowercase)
    df_clean['category'] = df_clean['category'].str.lower()
    df_clean['category'].fillna('bronze', inplace=True)

    print("\nCleaned data:")
    print(df_clean)

    print("\nData types after cleaning:")
    print(df_clean.dtypes)

    print("\nMissing values after cleaning:")
    print(df_clean.isnull().sum())

def exercise_8():
    """Feature engineering for time series"""
    print("\nExercise 8: Time Series Feature Engineering")
    print("-" * 40)

    # Create sales time series
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    np.random.seed(42)
    sales = np.random.randint(100, 500, 30)

    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales
    })

    print("Original time series (first 10 days):")
    print(df.head(10))

    # Extract datetime features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayName'] = df['Date'].dt.day_name()
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    print("\nWith datetime features:")
    print(df[['Date', 'Sales', 'DayName', 'Is_Weekend']].head(10))

    # Create lag features
    df['Sales_Lag1'] = df['Sales'].shift(1)
    df['Sales_Lag7'] = df['Sales'].shift(7)

    print("\nWith lag features:")
    print(df[['Date', 'Sales', 'Sales_Lag1', 'Sales_Lag7']].head(10))

    # Create rolling statistics
    df['Sales_MA7'] = df['Sales'].rolling(window=7).mean()
    df['Sales_MA7_Std'] = df['Sales'].rolling(window=7).std()

    print("\nWith rolling features (last 10 days):")
    print(df[['Date', 'Sales', 'Sales_MA7', 'Sales_MA7_Std']].tail(10).round(2))

    # Weekend vs weekday sales
    weekend_avg = df[df['Is_Weekend'] == 1]['Sales'].mean()
    weekday_avg = df[df['Is_Weekend'] == 0]['Sales'].mean()

    print(f"\nWeekend average sales: ${weekend_avg:.2f}")
    print(f"Weekday average sales: ${weekday_avg:.2f}")

def exercise_9():
    """Multi-column feature interactions"""
    print("\nExercise 9: Feature Interactions")
    print("-" * 40)

    # Create house price data
    df = pd.DataFrame({
        'SquareFeet': [1000, 1500, 2000, 2500, 3000],
        'Bedrooms': [2, 3, 3, 4, 4],
        'Bathrooms': [1, 2, 2, 3, 3],
        'Age': [10, 5, 15, 8, 3],
        'Price': [200000, 300000, 350000, 450000, 550000]
    })

    print("Original features:")
    print(df)

    # Create interaction features
    df['PricePerSqFt'] = df['Price'] / df['SquareFeet']
    df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']
    df['SqFtPerRoom'] = df['SquareFeet'] / df['TotalRooms']
    df['SqFt_Times_Rooms'] = df['SquareFeet'] * df['TotalRooms']

    # Polynomial features
    df['SqFt_Squared'] = df['SquareFeet'] ** 2
    df['Age_Squared'] = df['Age'] ** 2

    # Ratio features
    df['Bedroom_Bathroom_Ratio'] = df['Bedrooms'] / df['Bathrooms']

    # Age-based features
    df['Is_New'] = (df['Age'] < 5).astype(int)
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 5, 10, 20], labels=['New', 'Recent', 'Old'])

    print("\nWith interaction features:")
    print(df[['SquareFeet', 'TotalRooms', 'PricePerSqFt', 'SqFtPerRoom']].round(2))

    print("\nWith polynomial features:")
    print(df[['SquareFeet', 'SqFt_Squared', 'Age', 'Age_Squared']])

    print("\nWith categorical features:")
    print(df[['Age', 'Is_New', 'Age_Category']])

def exercise_10():
    """Advanced text features"""
    print("\nExercise 10: Advanced Text Features")
    print("-" * 40)

    # Product descriptions
    products = pd.DataFrame({
        'Product': ['Laptop Pro 15', 'Phone X', 'Tablet Mini 10', 'Laptop Air 13', 'Phone Pro Max'],
        'Description': ['High-end gaming laptop with powerful processor',
                       'Smartphone with advanced camera features',
                       'Compact tablet for reading and browsing',
                       'Lightweight laptop for students and professionals',
                       'Premium phone with best camera quality']
    })

    print("Product data:")
    print(products)

    # Extract text features
    products['Name_Length'] = products['Product'].str.len()
    products['Name_Words'] = products['Product'].str.split().str.len()
    products['Desc_Length'] = products['Description'].str.len()
    products['Desc_Words'] = products['Description'].str.split().str.len()

    # Pattern matching
    products['Has_Pro'] = products['Product'].str.contains('Pro', case=False).astype(int)
    products['Has_Number'] = products['Product'].str.contains(r'\d+', case=False).astype(int)
    products['Product_Type'] = products['Product'].str.extract(r'(Laptop|Phone|Tablet)', flags=re.IGNORECASE)

    # Extract numbers
    products['Screen_Size'] = products['Product'].str.extract(r'(\d+)').astype(float)

    # Keyword features
    products['Desc_Has_Gaming'] = products['Description'].str.contains('gaming', case=False).astype(int)
    products['Desc_Has_Camera'] = products['Description'].str.contains('camera', case=False).astype(int)
    products['Desc_Has_Premium'] = products['Description'].str.contains('premium|advanced|best', case=False).astype(int)

    print("\nText features:")
    print(products[['Product', 'Name_Length', 'Name_Words', 'Has_Pro', 'Product_Type']])

    print("\nExtracted features:")
    print(products[['Product', 'Screen_Size', 'Desc_Has_Premium']])

    print("\nDescription features:")
    print(products[['Product', 'Desc_Words', 'Desc_Has_Gaming', 'Desc_Has_Camera']])

def main():
    print("=" * 60)
    print("Data Cleaning Exercises")
    print("=" * 60)

    exercises = [
        exercise_1,
        exercise_2,
        exercise_3,
        exercise_4,
        exercise_5,
        exercise_6,
        exercise_7,
        exercise_8,
        exercise_9,
        exercise_10,
    ]

    for i, exercise in enumerate(exercises, 1):
        try:
            exercise()
        except Exception as e:
            print(f"\nError in exercise {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Additional Practice:")
    print("1. Build end-to-end pipeline: load -> clean -> engineer -> scale")
    print("2. Handle mixed data types in single column")
    print("3. Create custom imputation strategy based on correlations")
    print("4. Implement automatic outlier detection across all numeric columns")
    print("5. Build text preprocessing for social media data (emojis, hashtags)")
    print("6. Create time-based aggregation features (hourly, daily, weekly)")
    print("7. Engineer polynomial interactions for all numeric features")
    print("8. Build custom binning strategy based on domain knowledge")
    print("9. Implement encoding for high-cardinality categorical features")
    print("10. Create composite features from multiple data sources")
    print("=" * 60)

if __name__ == "__main__":
    main()
