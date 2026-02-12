import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve)
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ChurnPredictor:
    def __init__(self):
        """Load and merge all datasets"""
        print("Loading customer datasets...")
        
        self.demo = pd.read_csv('data/customer_demographics.csv')
        self.trans = pd.read_csv('data/customer_transactions.csv')
        self.support = pd.read_csv('data/customer_support.csv')
        self.survey = pd.read_csv('data/customer_survey.csv')
        self.churn = pd.read_csv('data/customer_churn.csv')
        
        # Merge into single DataFrame
        self.df = self.demo.merge(self.trans, on='customer_id', how='left')
        self.df = self.df.merge(self.support, on='customer_id', how='left')
        self.df = self.df.merge(self.survey, on='customer_id', how='left')
        self.df = self.df.merge(self.churn, on='customer_id', how='left')
        
        # Create derived features
        self.df['avg_monthly_spend'] = self.df['total_spend'] / self.df['tenure_months']
        self.df['transaction_frequency'] = self.df['total_transactions'] / self.df['tenure_months']
        
        # Feature engineering
        self.df['total_satisfaction'] = (self.df['satisfaction_product'] + 
                                         self.df['satisfaction_service'] + 
                                         self.df['satisfaction_price']) / 3
        
        self.df['is_high_value'] = (self.df['total_spend'] > self.df['total_spend'].median()).astype(int)
        
        # Encode categorical variables
        self.le_dict = {}
        cat_cols = ['gender', 'region', 'preferred_category', 'preferred_channel']
        for col in cat_cols:
            le = LabelEncoder()
            self.df[col+'_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.le_dict[col] = le
        
        print(f"Data merged. Shape: {self.df.shape}")
        print(f"Churn rate: {self.df['churned'].mean():.2%}")
    
    def exploratory_analysis(self):
        """Perform EDA and visualize key insights"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Customer Churn - Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Churn distribution
        churn_counts = self.df['churned'].value_counts()
        axes[0,0].pie(churn_counts, labels=['Active', 'Churned'], autopct='%1.1f%%',
                      colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0,0].set_title('Churn Distribution')
        
        # 2. Tenure vs Churn
        sns.boxplot(data=self.df, x='churned', y='tenure_months', ax=axes[0,1])
        axes[0,1].set_title('Tenure Months by Churn Status')
        axes[0,1].set_xlabel('Churned')
        axes[0,1].set_ylabel('Tenure (months)')
        
        # 3. Satisfaction vs Churn
        sns.boxplot(data=self.df, x='churned', y='total_satisfaction', ax=axes[0,2])
        axes[0,2].set_title('Satisfaction Score by Churn Status')
        axes[0,2].set_xlabel('Churned')
        axes[0,2].set_ylabel('Total Satisfaction')
        
        # 4. Support tickets vs Churn
        sns.boxplot(data=self.df, x='churned', y='support_tickets', ax=axes[1,0])
        axes[1,0].set_title('Support Tickets by Churn Status')
        axes[1,0].set_xlabel('Churned')
        axes[1,0].set_ylabel('Number of Tickets')
        
        # 5. Average order value vs Churn
        sns.boxplot(data=self.df, x='churned', y='avg_order_value', ax=axes[1,1])
        axes[1,1].set_title('Average Order Value by Churn Status')
        axes[1,1].set_xlabel('Churned')
        axes[1,1].set_ylabel('Avg Order Value ($)')
        
        # 6. Correlation heatmap of numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                    center=0, square=True, ax=axes[1,2])
        axes[1,2].set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('churn_eda.png', dpi=300)
        plt.show()
        
        # Print summary statistics by churn group
        print("\nSummary Statistics by Churn Status:")
        print(self.df.groupby('churned')[['tenure_months', 'total_spend', 
                                          'support_tickets', 'total_satisfaction']].mean())
    
    def prepare_features(self):
        """Prepare feature matrix and target variable"""
        # Define features
        feature_cols = [
            'age', 'gender_encoded', 'annual_income', 'tenure_months',
            'region_encoded', 'has_partner', 'has_dependents',
            'total_transactions', 'avg_order_value', 'total_spend',
            'recency_days', 'preferred_category_encoded',
            'support_tickets', 'avg_resolution_hours', 'complaints',
            'preferred_channel_encoded',
            'satisfaction_product', 'satisfaction_service', 'satisfaction_price',
            'nps_score', 'recommend_score',
            'avg_monthly_spend', 'transaction_frequency', 'total_satisfaction',
            'is_high_value'
        ]
        
        X = self.df[feature_cols].copy()
        y = self.df['churned'].copy()
        
        # Handle missing values (none in synthetic data, but just in case)
        X = X.fillna(X.median())
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple classifiers and evaluate performance"""
        print("\n" + "="*60)
        print("MACHINE LEARNING MODEL TRAINING")
        print("="*60)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, 
                                          random_state=42, eval_metric='logloss')
        }
        
        results = {}
        self.models = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:,1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:,1]
            
            self.models[name] = model
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': auc
            }
        
        # Display results
        results_df = pd.DataFrame(results).T
        print("\nModel Performance Comparison:")
        print(results_df.round(4))
        
        # Cross-validation
        print("\nCross-Validation Scores (ROC-AUC):")
        for name, model in models.items():
            if name == 'Logistic Regression':
                scores = cross_val_score(model, X_train_scaled, y_train, 
                                         cv=5, scoring='roc_auc')
            else:
                scores = cross_val_score(model, X_train, y_train, 
                                         cv=5, scoring='roc_auc')
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        # Store test data for later
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        return results_df, (X_train, X_test, y_train, y_test)
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(8,6))
        
        for name, model in self.models.items():
            if name == 'Logistic Regression':
                y_proba = model.predict_proba(self.X_test_scaled)[:,1]
            else:
                y_proba = model.predict_proba(self.X_test)[:,1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0,1], [0,1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300)
        plt.show()
    
    def feature_importance_analysis(self, model_name='Random Forest'):
        """Analyze and plot feature importance"""
        model = self.models.get(model_name)
        if model is None:
            print(f"Model {model_name} not found.")
            return
        
        # Get feature names
        feature_cols = self.prepare_features()[0].columns
        
        if model_name == 'Logistic Regression':
            # Use absolute coefficients as importance
            importances = np.abs(model.coef_[0])
        else:
            importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        # Plot
        plt.figure(figsize=(10,6))
        sns.barplot(data=importance_df, y='feature', x='importance', 
                    palette='viridis')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()
        
        print(f"\nTop 10 Important Features ({model_name}):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def customer_segmentation(self):
        """Cluster customers based on risk and value"""
        print("\n" + "="*60)
        print("CUSTOMER SEGMENTATION FOR CHURN RISK")
        print("="*60)
        
        # Select features for segmentation
        seg_features = ['tenure_months', 'total_spend', 'support_tickets', 
                        'total_satisfaction', 'recency_days']
        
        X_seg = self.df[seg_features].copy()
        X_seg_scaled = StandardScaler().fit_transform(X_seg)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df['segment'] = kmeans.fit_predict(X_seg_scaled)
        
        # Profile segments
        segment_profile = self.df.groupby('segment').agg({
            'customer_id': 'count',
            'tenure_months': 'mean',
            'total_spend': 'mean',
            'support_tickets': 'mean',
            'total_satisfaction': 'mean',
            'recency_days': 'mean',
            'churned': 'mean'
        }).round(2)
        
        print("\nCustomer Segments Profile:")
        print(segment_profile)
        
        # Rename segments for interpretation
        segment_names = {}
        for seg in segment_profile.index:
            churn_rate = segment_profile.loc[seg, 'churned']
            if churn_rate > 0.5:
                name = 'High Risk - Churn Prone'
            elif churn_rate > 0.2:
                name = 'Medium Risk'
            else:
                name = 'Low Risk - Loyal'
            segment_names[seg] = name
        
        self.df['segment_name'] = self.df['segment'].map(segment_names)
        
        # Visualize segments
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=self.df, x='total_spend', y='tenure_months', 
                        hue='segment_name', style='churned', s=100, alpha=0.7)
        plt.title('Customer Segments by Spend and Tenure')
        plt.xlabel('Total Spend ($)')
        plt.ylabel('Tenure (months)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('customer_segments.png', dpi=300)
        plt.show()
        
        return segment_profile
    
    def generate_insights_report(self):
        """Produce actionable insights and recommendations"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RETENTION STRATEGIES")
        print("="*60)
        
        # Churn drivers summary
        churned = self.df[self.df['churned'] == 1]
        active = self.df[self.df['churned'] == 0]
        
        insights = []
        insights.append("Key Churn Drivers:")
        insights.append(f"- Tenure: Churned customers have {churned['tenure_months'].mean():.1f} months tenure vs {active['tenure_months'].mean():.1f} for active")
        insights.append(f"- Satisfaction: Churned satisfaction {churned['total_satisfaction'].mean():.2f} vs {active['total_satisfaction'].mean():.2f}")
        insights.append(f"- Support Tickets: Churned avg {churned['support_tickets'].mean():.2f} vs {active['support_tickets'].mean():.2f}")
        insights.append(f"- Recency: Churned last purchase {churned['recency_days'].mean():.1f} days ago vs {active['recency_days'].mean():.1f}")
        
        insights.append("\nHigh-Risk Segment Characteristics:")
        high_risk = self.df[self.df['segment_name'].str.contains('High Risk')]
        if len(high_risk) > 0:
            insights.append(f"- Size: {len(high_risk)} customers ({len(high_risk)/len(self.df)*100:.1f}%)")
            insights.append(f"- Avg Spend: ${high_risk['total_spend'].mean():.2f}")
            insights.append(f"- Churn Rate: {high_risk['churned'].mean():.2%}")
        
        insights.append("\nRecommended Retention Actions:")
        insights.append("1. Proactive outreach to customers with satisfaction < 5 and high recency")
        insights.append("2. Offer loyalty discounts to short-tenure customers (<12 months)")
        insights.append("3. Reduce support resolution time for high-ticket customers")
        insights.append("4. Implement win-back campaigns for customers who churned within 30 days")
        insights.append("5. Personalize communication based on preferred channel and category")
        
        print("\n".join(insights))
        
        # Save to file
        with open('churn_insights.txt', 'w') as f:
            f.write("\n".join(insights))
    
    def run_full_pipeline(self):
        """Execute complete analysis pipeline"""
        print("\n" + "="*60)
        print("CUSTOMER CHURN PREDICTION - FULL PIPELINE")
        print("="*60)
        
        # 1. EDA
        self.exploratory_analysis()
        
        # 2. Feature preparation
        X, y = self.prepare_features()
        
        # 3. Model training
        results, split_data = self.train_models(X, y)
        
        # 4. ROC curves
        self.plot_roc_curves()
        
        # 5. Feature importance
        self.feature_importance_analysis('Random Forest')
        self.feature_importance_analysis('XGBoost')
        
        # 6. Customer segmentation
        self.customer_segmentation()
        
        # 7. Insights report
        self.generate_insights_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Outputs saved:")
        print("- churn_eda.png")
        print("- roc_curves.png")
        print("- feature_importance.png")
        print("- customer_segments.png")
        print("- churn_insights.txt")

def main():
    predictor = ChurnPredictor()
    predictor.run_full_pipeline()

if __name__ == '__main__':
    main()
