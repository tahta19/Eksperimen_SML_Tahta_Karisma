import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset from the specified file path."""
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data):
    """Handle missing values by imputing with the mean (if any)."""
    imputer = SimpleImputer(strategy='mean')
    data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = imputer.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    return data

def remove_duplicates(data):
    """Remove duplicate rows if any."""
    data.drop_duplicates(inplace=True)
    return data

def normalize_data(data):
    """Normalize the relevant numeric columns."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    return scaled_data

def handle_outliers(data):
    """Handle outliers by capping values."""
    data['Annual Income (k$)'] = np.where(data['Annual Income (k$)'] > 120, 120, data['Annual Income (k$)'])
    return data

def encode_gender(data):
    """Encode the 'Gender' column."""
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    return data

def save_preprocessed_data(data, output_path):
    """Save the preprocessed data to a CSV file."""
    data.to_csv(output_path, index=False)


def preprocess_data(input_file, output_file):
    """Run the entire preprocessing pipeline."""
    data = load_data(input_file)  
    data = handle_missing_values(data)  
    data = remove_duplicates(data)  
    data = handle_outliers(data) 
    data = encode_gender(data) 
    scaled_data = normalize_data(data)  
    
    save_preprocessed_data(data, output_file)
    
    return scaled_data

if __name__ == "__main__":
    input_file = 'Mall_Customers.csv'  
    output_file = 'Mall_Customers_Preprocessed.csv'
    
    scaled_data = preprocess_data(input_file, output_file)
    print(f"Preprocessing completed. Processed data saved to {output_file}.")
