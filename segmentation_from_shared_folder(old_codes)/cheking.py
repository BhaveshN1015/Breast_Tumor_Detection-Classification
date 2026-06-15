import pandas as pd

EXCEL_FILE = r"D:\Breast_Tumor_AI_Project\TCIA-Breastdx.xlsx"
OUTPUT_CSV = r"D:\Breast_Tumor_AI_Project\clean_labels.csv"

# Load Excel
df = pd.read_excel(EXCEL_FILE, header=None)

# ===============================
# 🔍 Find pathology row
# ===============================
pathology_row_idx = None

for i in range(len(df)):
    row = df.iloc[i].astype(str).str.lower()
    if "pathology" in " ".join(row):
        pathology_row_idx = i
        break

if pathology_row_idx is None:
    print("❌ Pathology row not found")
    exit()

print(f"✅ Pathology row: {pathology_row_idx}")

# ===============================
# 🔍 Find patient columns
# ===============================
patient_columns = []

for col in df.columns:
    for row in range(len(df)):
        cell = str(df.iloc[row, col]).strip()

        if "breastdx" in cell.lower():
            patient_columns.append((col, cell))
            break

print(f"✅ Found {len(patient_columns)} patients")

# ===============================
# 🏷 Label cleaning
# ===============================
def clean_label(x):
    x = str(x).lower()

    # malignant keywords
    if any(k in x for k in ["carcinoma", "invasive", "idc", "cancer"]):
        return "Malignant"

    # benign keywords
    elif any(k in x for k in ["benign", "fibroadenoma"]):
        return "Benign"

    # unknown / missing
    elif x.strip() == "" or x == "nan":
        return "Normal"

    else:
        return "Unknown"

# ===============================
# 📊 Extract all data
# ===============================
clean_data = []

for col, patient in patient_columns:
    label_raw = df.iloc[pathology_row_idx, col]
    label = clean_label(label_raw)

    clean_data.append({
        "patient_id": patient.strip(),
        "label": label
    })

clean_df = pd.DataFrame(clean_data)

# ===============================
# 💾 Save
# ===============================
clean_df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ FULL dataset saved!")
print("\n📊 Sample:\n")
print(clean_df.head())
print("\n📊 Total patients:", len(clean_df))