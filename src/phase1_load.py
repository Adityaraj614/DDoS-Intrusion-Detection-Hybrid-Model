# phase1_load.py

import pandas as pd
import os

# Define dataset path
DATA_PATH = os.path.join("data")

train_file = os.path.join(DATA_PATH, "KDDTrain+.txt")
test_file = os.path.join(DATA_PATH, "KDDTest+.txt")

# Column names (official NSL-KDD format)
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label'
]

# Load datasets
train_df = pd.read_csv(train_file, names=columns + ['difficulty'])
test_df = pd.read_csv(test_file, names=columns + ['difficulty'])

# Drop difficulty column
train_df = train_df.drop('difficulty', axis=1)
test_df = test_df.drop('difficulty', axis=1)

# Basic info
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nSample Data:")
print(train_df.head())

print("\nLabel Distribution:")
print(train_df['label'].value_counts())