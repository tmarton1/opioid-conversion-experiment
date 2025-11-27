import os
import shutil

# Create 'data' and 'figures' folders if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('figures'):
    os.makedirs('figures')

# Move .csv files to 'data'
for filename in os.listdir('.'):
    if filename.endswith('.csv') and os.path.isfile(filename):
        shutil.move(filename, os.path.join('data', filename))
        print(f"Moved to data: {filename}")

# Move .html and .pdf files to 'figures'
for filename in os.listdir('.'):
    if (filename.endswith('.html') or filename.endswith('.pdf')) and os.path.isfile(filename):
        shutil.move(filename, os.path.join('figures', filename))
        print(f"Moved to figures: {filename}")