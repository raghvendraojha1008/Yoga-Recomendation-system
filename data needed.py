import os
import pandas as pd

# --- CONFIGURATION ---
# PLEASE DOUBLE CHECK THESE PATHS
yoga_data_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Poses Recommendation\Yoga Data.xlsx"
image_base_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Poses Classification"
output_file = "yoga_data_inventory.txt"

def collect_system_data():
    report = []
    folders = [] # Initialize empty to prevent UnboundLocalError
    report.append("=== YOGA PROJECT DATA INVENTORY REPORT ===\n")

    # 1. SCAN EXCEL DATA
    try:
        df_yoga = pd.read_excel(yoga_data_path)
        report.append(f"Total Asanas in Excel: {len(df_yoga)}")
        
        # Collect unique Target Area IDs (e.g., 18, 31)
        target_areas = sorted(df_yoga['Target Areas'].unique().tolist())
        report.append(f"Unique Target Area IDs found: {target_areas}")
        report.append("Action: These IDs need to be mapped to body parts (e.g., Core, Back).\n")
    except Exception as e:
        report.append(f"CRITICAL ERROR reading Excel: {e}")
        df_yoga = None

    # 2. SCAN IMAGE FOLDERS
    if os.path.exists(image_base_path):
        try:
            folders = [f for f in os.listdir(image_base_path) if os.path.isdir(os.path.join(image_base_path, f))]
            report.append(f"Total Image Folders found: {len(folders)}")
            report.append("List of available folders:")
            for folder in sorted(folders):
                report.append(f" - {folder}")
        except Exception as e:
            report.append(f"ERROR reading Image Directory: {e}")
    else:
        report.append(f"ERROR: Image path does not exist: {image_base_path}")

    # 3. IDENTIFY MISSING LINKS
    report.append("\n=== POTENTIAL NAME MISMATCHES ===")
    if df_yoga is not None and len(folders) > 0:
        asana_names = df_yoga['AName'].tolist()
        for name in asana_names:
            # Check if name (lowercase) exists in any folder name
            match_found = any(str(name).lower().replace(" ", "-") in f.lower().replace("_", "-") for f in folders)
            if not match_found:
                report.append(f"MISSING IMAGE MAPPING: '{name}' does not have a matching folder.")
    elif len(folders) == 0:
        report.append("SKIPPING MATCH CHECK: No image folders were found to compare against.")

    # SAVE TO TEXT FILE
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"Inventory complete! Please check '{output_file}' in your project folder.")

if __name__ == "__main__":
    collect_system_data()
