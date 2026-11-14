import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load your data
df = pd.read_excel('output.xlsx', sheet_name='output')


# 2. Function to extract Base Name and Shape Type from the 'Audio' column
def parse_audio_name(name):
  # Remove trailing sample rates or suffixes if needed, e.g., '_11025'
  base = name.split('_11025')[0]

  if base.endswith('Cl'):
    person = base[:-2]  # Remove 'Cl'
    shape_type = 'Cl'
  elif base.endswith('CS'):
    person = base[:-2]  # Remove 'CS'
    shape_type = 'CS'
  else:
    person = base
    shape_type = 'Normal'

  return person, shape_type


# Apply parsing
df[['Person', 'ShapeType']] = df['Audio'].apply(
    lambda x: pd.Series(parse_audio_name(x))
)

# 3. Setup the plot
plt.figure(figsize=(12, 6))

# Define distinct shapes for Normal, Cl, and CS
markers = {'Normal': 'o', 'Cl': '^', 'CS': 's'}  # circle, triangle, square

# Create scatter plot with Seaborn
# hue maps to 'Person' (same color for same base name)
# style maps to 'ShapeType' (different shape for Cl/CS vs Normal)
sns.scatterplot(
    data=df,
    x='Condition',
    y='Confidence',
    hue='Person',
    style='ShapeType',
    markers=markers,
    palette='tab10',  # Color palette for speakers
    s=100,
    alpha=0.8,
)

plt.title('Confidence by Condition (Grouped by Speaker & Suffix Type)')
plt.xlabel('Condition')
plt.ylabel('Confidence')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig("excel_data.png")
