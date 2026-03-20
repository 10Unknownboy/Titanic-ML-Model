import pandas as pd
import glob
import os

def main():
    # Find all submission files in the current directory
    files = glob.glob("submission*.csv")
    
    if len(files) < 2:
        print(f"Found only {len(files)} submission file(s). Need at least 2 to compare.")
        if files:
            print(f"File found: {files[0]}")
        return

    # Sort files by their modification time (oldest first, newest last)
    files.sort(key=os.path.getmtime)
    
    # Select the two most recent files
    f1 = files[-2]
    f2 = files[-1]

    print(f"Comparing the two most recent submission files:\n 1. {f1}\n 2. {f2}\n")

    # Load all predictions
    try:
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
    except Exception as e:
        print(f"Error reading files: {e}")
        return
        
    if "Survived" not in df1.columns or "Survived" not in df2.columns:
        print("Warning: One of the files does not have a 'Survived' column.")
        return
        
    preds1 = df1["Survived"].values
    preds2 = df2["Survived"].values

    print("Agreement (%):")
    print("-" * 60)
    
    if len(preds1) != len(preds2):
        print(f"Cannot compare due to row count mismatch ({len(preds1)} vs {len(preds2)}).")
        return
        
    # Calculate agreement percentage and absolute number of differences
    agreement = (preds1 == preds2).mean() * 100
    differences = (preds1 != preds2).sum()
    
    print(f"{f1} vs {f2}:")
    print(f"  → {agreement:.2f}% agreement ({differences} different predictions)\n")

if __name__ == "__main__":
    main()
