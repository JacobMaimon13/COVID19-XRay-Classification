import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob

def prepare_data(data_dir='data'):
    print("--- Preparing Data ---")
    
    # 1. Define Paths (Assuming datasets are extracted in 'data/')
    # You might need to adjust these paths depending on how you extract the zips
    covid_images_path = Path(data_dir) / "covid-chestxray-dataset" / "images"
    covid_metadata_path = Path(data_dir) / "covid-chestxray-dataset" / "metadata.csv"
    
    pneumonia_base_path = Path(data_dir) / "chest_xray"
    
    # 2. Process COVID Dataset
    if not covid_metadata_path.exists():
        print(f"Error: COVID metadata not found at {covid_metadata_path}")
        return

    covid_metadata = pd.read_csv(covid_metadata_path)
    
    # Filter PA views
    covid_df = covid_metadata[covid_metadata['view'] == 'PA'].copy()
    
    # Create full filepath
    covid_df['filepath'] = covid_df['filename'].apply(
        lambda x: str(covid_images_path / x)
    )
    
    # Filter only COVID-19 findings
    covid_df = covid_df[covid_df['finding'].str.contains('COVID-19', case=False, na=False)]
    
    # Standardize columns
    covid_df = covid_df[['finding', 'filepath']]
    covid_df['label'] = 2 # COVID-19
    
    print(f"Found {len(covid_df)} COVID-19 PA images.")

    # 3. Process Pneumonia/Normal Dataset
    def create_labeled_df(folder_path):
        data = []
        if not os.path.exists(folder_path):
            print(f"Warning: Path not found {folder_path}")
            return pd.DataFrame(columns=['finding', 'filepath', 'label'])
            
        for label_name in os.listdir(folder_path):
            label_dir = os.path.join(folder_path, label_name)
            if not os.path.isdir(label_dir): continue
            
            for filename in os.listdir(label_dir):
                if not filename.lower().endswith(('.jpeg', '.jpg', '.png')): continue
                
                filepath = os.path.join(label_dir, filename)
                
                if label_name == 'PNEUMONIA':
                    if 'bacteria' in filename.lower():
                        data.append(('BACTERIA', filepath, 1))
                elif label_name == 'NORMAL':
                    data.append(('NORMAL', filepath, 0))
        
        return pd.DataFrame(data, columns=['finding', 'filepath', 'label'])

    train_path = pneumonia_base_path / "train"
    test_path = pneumonia_base_path / "test"
    
    pneumonia_train = create_labeled_df(train_path)
    pneumonia_test = create_labeled_df(test_path)
    
    print(f"Found {len(pneumonia_train) + len(pneumonia_test)} Pneumonia/Normal images.")

    # 4. Combine and Split
    full_df = pd.concat([covid_df, pneumonia_train, pneumonia_test], ignore_index=True)
    
    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split to Train/Test (80/20)
    train_df, test_df = train_test_split(
        full_df, test_size=0.2, stratify=full_df['label'], random_state=42
    )
    
    # 5. Save to CSV
    train_csv_path = os.path.join(data_dir, 'train.csv')
    test_csv_path = os.path.join(data_dir, 'test.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Successfully saved:\n- {train_csv_path} ({len(train_df)} samples)\n- {test_csv_path} ({len(test_df)} samples)")

if __name__ == "__main__":
    prepare_data()
