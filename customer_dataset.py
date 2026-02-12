import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def create_demographics_data():
    """Customer demographics (50 customers)"""
    n = 50
    customer_ids = [f'C{1000+i}' for i in range(n)]
    
    ages = np.random.randint(18, 70, n)
    genders = np.random.choice(['Male', 'Female'], n, p=[0.5, 0.5])
    income = np.random.normal(60000, 20000, n).astype(int)
    income = np.clip(income, 25000, 150000)
    
    # Tenure in months
    tenure = np.random.randint(1, 72, n)
    
    # Location
    regions = np.random.choice(['North', 'South', 'East', 'West'], n)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'annual_income': income,
        'tenure_months': tenure,
        'region': regions,
        'has_partner': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'has_dependents': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    return df

def create_transaction_data():
    """Aggregated transaction history per customer (50 rows)"""
    n = 50
    customer_ids = [f'C{1000+i}' for i in range(n)]
    
    # Number of transactions per customer
    num_transactions = np.random.randint(1, 50, n)
    
    # Average order value
    avg_order_value = np.random.normal(100, 30, n).round(2)
    avg_order_value = np.clip(avg_order_value, 20, 200)
    
    # Total spend
    total_spend = num_transactions * avg_order_value * np.random.uniform(0.9, 1.1, n)
    total_spend = total_spend.round(2)
    
    # Recency (days since last purchase)
    recency_days = np.random.randint(1, 180, n)
    
    # Product categories purchased
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    preferred_category = np.random.choice(categories, n)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'total_transactions': num_transactions,
        'avg_order_value': avg_order_value,
        'total_spend': total_spend,
        'recency_days': recency_days,
        'preferred_category': preferred_category
    })
    return df

def create_support_interaction_data():
    """Customer support interactions aggregated (50 customers)"""
    n = 50
    customer_ids = [f'C{1000+i}' for i in range(n)]
    
    # Number of support tickets
    tickets = np.random.poisson(2, n)
    tickets = np.clip(tickets, 0, 10)
    
    # Average resolution time (hours)
    avg_resolution_hours = np.random.gamma(2, 4, n).round(2)
    avg_resolution_hours = np.clip(avg_resolution_hours, 0.5, 48)
    
    # Number of complaints
    complaints = np.random.poisson(0.5, n).astype(int)
    complaints = np.clip(complaints, 0, 5)
    
    # Interaction channels
    channel = np.random.choice(['Email', 'Chat', 'Phone', 'Self-Service'], n)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'support_tickets': tickets,
        'avg_resolution_hours': avg_resolution_hours,
        'complaints': complaints,
        'preferred_channel': channel
    })
    return df

def create_survey_data():
    """Customer satisfaction survey scores (50 customers)"""
    n = 50
    customer_ids = [f'C{1000+i}' for i in range(n)]
    
    # Satisfaction scores (1-10)
    satisfaction_product = np.random.randint(1, 11, n)
    satisfaction_service = np.random.randint(1, 11, n)
    satisfaction_price = np.random.randint(1, 11, n)
    
    # NPS (0-10)
    nps_score = np.random.randint(0, 11, n)
    
    # Recommend likelihood (1-10)
    recommend_score = np.random.randint(1, 11, n)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'satisfaction_product': satisfaction_product,
        'satisfaction_service': satisfaction_service,
        'satisfaction_price': satisfaction_price,
        'nps_score': nps_score,
        'recommend_score': recommend_score
    })
    return df

def create_churn_data():
    """Churn labels (50 customers)"""
    n = 50
    customer_ids = [f'C{1000+i}' for i in range(n)]
    
    # Churn flag: 1 = churned, 0 = active
    # We'll make churn correlated with tenure, satisfaction, etc.
    # But here we create random with 20-30% churn rate
    churn = np.random.choice([0, 1], n, p=[0.75, 0.25])
    
    # Churn date (if churned)
    churn_date = []
    for i in range(n):
        if churn[i] == 1:
            days_ago = np.random.randint(1, 180)
            date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        else:
            date = ''
        churn_date.append(date)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'churned': churn,
        'churn_date': churn_date
    })
    return df

def main():
    import os
    os.makedirs('data', exist_ok=True)
    
    demo_df = create_demographics_data()
    demo_df.to_csv('data/customer_demographics.csv', index=False)
    print(f"Created customer_demographics.csv with {len(demo_df)} rows")
    
    trans_df = create_transaction_data()
    trans_df.to_csv('data/customer_transactions.csv', index=False)
    print(f"Created customer_transactions.csv with {len(trans_df)} rows")
    
    support_df = create_support_interaction_data()
    support_df.to_csv('data/customer_support.csv', index=False)
    print(f"Created customer_support.csv with {len(support_df)} rows")
    
    survey_df = create_survey_data()
    survey_df.to_csv('data/customer_survey.csv', index=False)
    print(f"Created customer_survey.csv with {len(survey_df)} rows")
    
    churn_df = create_churn_data()
    churn_df.to_csv('data/customer_churn.csv', index=False)
    print(f"Created customer_churn.csv with {len(churn_df)} rows")
    
    print("\nAll datasets created successfully!")

if __name__ == '__main__':
    main()
