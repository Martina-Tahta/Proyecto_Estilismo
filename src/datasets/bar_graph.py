import os 
import re
import pandas as pd
import matplotlib.pyplot as plt

raw_dataset_path = '/home/martina/2025/investigacion/Proyecto_Estilismo/data/raw/SeasonsModel'

image_files = []
season_list = []
category_list = []
count = {'spring': {}, 'winter': {}, 'summer': {}, 'autumn': {}}

for image_file in os.listdir(raw_dataset_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        season_match = re.match(r'([a-zA-Z]+_[a-zA-Z]+)_\d+', image_file)
        if season_match:
            #print(season_match.group(1))
            season_cat = season_match.group(1)
            print(season_cat)
            category, season = season_cat.split('_', 1) 
            if category in count[season]:
                count[season][category] += 1
            else:
                count[season][category] = 1
            
            

data = []
for season, cats in count.items():
    for category, num in cats.items():
        data.append({'season': season.capitalize(), 'category': category, 'count': num})

df = pd.DataFrame(data)

# Crear gráfico como en tu ejemplo
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

for ax, season in zip(axes, seasons):
    subset = df[df['season'] == season]
    ax.bar(subset['category'], subset['count'])
    ax.set_title(season)
    ax.set_xlabel('Category')
    if season == 'Spring':
        ax.set_ylabel('Samples')

plt.tight_layout()
plt.savefig('/home/martina/2025/investigacion/Proyecto_Estilismo/src/datasets/bar_graph_seasonsModel.png')
plt.show()