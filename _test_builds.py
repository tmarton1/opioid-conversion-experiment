import traceback
from visualize.comprehensive_figures import ComprehensiveFigureBuilder

fb = ComprehensiveFigureBuilder('results')
for name, method_name, title, desc in fb.FIGURE_METADATA:
    try:
        fig = getattr(fb, method_name)()
        print(f'OK   {name}')
    except Exception as e:
        print(f'FAIL {name}: {e}')
        traceback.print_exc()
