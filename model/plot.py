import matplotlib.pyplot as plt
import pandas as pd

# 1. Load your data
df = pd.read_excel('output.xlsx', sheet_name='output')


# 2. Function to extract Base Name and Shape Type from the 'Audio' column
def parse_audio_name(name):
  base = name.split('_11025')[0]
  if base.endswith('Cl'):
    person = base[:-2]
    shape_type = 'Cl'
  elif base.endswith('CS'):
    person = base[:-2]
    shape_type = 'CS'
  else:
    person = base
    shape_type = 'Normal'
  return person, shape_type


df[['Person', 'ShapeType']] = df['Audio'].apply(
    lambda x: pd.Series(parse_audio_name(x))
)

# 3. Setup the plot
plt.figure(figsize=(14, 7))

# Get unique persons and assign a color to each using a colormap
persons = df['Person'].unique()
colors = plt.get_cmap('tab20', len(persons))
person_color_map = {
    person: colors(i) for i, person in enumerate(persons)
}

# Define markers for each shape type
markers = {'Normal': 'o', 'Cl': '^', 'CS': 's'}

# Map string categories on X-axis to categorical codes or positions to handle strings properly
conditions = df['Condition'].unique()
condition_map = {cond: i for i, cond in enumerate(conditions)}
df['Cond_Code'] = df['Condition'].map(condition_map)

# Plot point by point or group by group
for person in persons:
  for shape_type, marker in markers.items():
    subset = df[(df['Person'] == person) & (df['ShapeType'] == shape_type)]
    if not subset.empty:
      plt.scatter(
          subset['Cond_Code'],
          subset['Confidence'],
          color=person_color_map[person],
          marker=marker,
          s=80,
          alpha=0.8,
          label=f'{person} ({shape_type})'
          if shape_type != 'Normal'
          else person,
      )

# Customize X-axis labels
plt.xticks(list(condition_map.values()), list(condition_map.keys()), rotation=45)
plt.title(
    'Confidence by Condition (Grouped by Speaker Color & Suffix Shape)',
    fontsize=14,
)
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Confidence', fontsize=12)

# Place legend outside to keep it clean
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
plt.tight_layout()
plt.show()
