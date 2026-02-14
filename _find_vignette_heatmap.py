import json
with open('opioid-conversion.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'vignette' in src and ('Heatmap' in src or 'heatmap' in src):
        print(f'Cell {i}: {src[:300]}')
        print('---')
    # Also check for the output file names
    for out in cell.get('outputs', []):
        text = ''.join(out.get('text', []))
        if 'vignette' in text and 'heatmap' in text:
            print(f'Cell {i} OUTPUT: {text[:200]}')
            print('---')
