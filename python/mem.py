import psutil

# Function to display memory usage
def display_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024**3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%\n")

# Example usage: check memory before data processing
print("Memory usage before loading data:")
display_memory_usage()

# Your data loading step
df = pd.read_csv('../data/creditcard.csv')

print("Memory usage after loading data:")
display_memory_usage()

# After resampling with SMOTE
X_smote, y_smote = smote.fit_resample(X, y)
print("Memory usage after SMOTE resampling:")
display_memory_usage()

# After scaling
X_standardized = scaler.fit_transform(X_smote)
print("Memory usage after scaling:")
display_memory_usage()
#----------------------------------------------------------test1-------------------------------> out put --->

