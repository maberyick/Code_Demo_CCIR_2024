from fmcib.run import get_features
import pandas as pd
from fmcib.visualization.verify_io import visualize_seed_point
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Path to your CSV file
csv_file_path = "/mnt/Data1/GitHub/Code_Demo_CCIR_2024/script_code_5/demo_image/demo_file.csv"

# Load CSV to verify paths
df = pd.read_csv(csv_file_path)
print("CSV Data:", df)

# Extract features using the get_features function
try:
    feature_df = get_features(csv_file_path)
    # Print or save the extracted features
    print(feature_df)
    # Visualize the first seed point for verification
    visualize_seed_point(feature_df.iloc[0])
except Exception as e:
    logging.error("Error in feature extraction: ", exc_info=e)
