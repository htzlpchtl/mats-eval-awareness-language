"""Shared, editable dataset diagram for the report and figure bundle."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_dataset_design():
    ink, grey = '#233140', '#65717C'
    fig = plt.figure(figsize=(11, 4.15), facecolor='white')
    for x, text in [(.363, 'Benchmark Format'), (.782, 'Casual Format')]:
        fig.text(x, .945, text, fontsize=14, ha='center', va='center',
                 color=grey, fontfamily='DejaVu Sans')
    for y, text in [(.68, 'Eval Class'), (.255, 'Deploy Class')]:
        fig.text(.083, y, text, fontsize=12.5, ha='center', va='center',
                 color=ink, fontfamily='DejaVu Sans')
    cells = [
        (.17, .495, '#B65B23', 'Benchmark Eval', [
            'Knowledge across subjects (MMLU)',
            'Science questions (ARC)',
            'Grade School Math (GSM8K)']),
        (.589, .495, '#865CA6', 'Casual Eval', [
            'MuTual: a dialogue-reasoning benchmark.',
            'We use only the first turn of each dialogue,',
            'so the prompt looks like everyday conversation.']),
        (.17, .07, '#246E92', 'Benchmark Deploy', [
            'Rewritten ShareGPT',
            'User requests rewritten to look like',
            'standard benchmark questions.']),
        (.589, .07, '#167B73', 'Casual Deploy', [
            'ShareGPT',
            'Original first-turn user requests',
            'from conversations with ChatGPT.']),
    ]
    for left, bottom, colour, heading, lines in cells:
        ax = fig.add_axes([left, bottom, .386, .37])
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((.002, .005), .996, .99,
            boxstyle='round,pad=0,rounding_size=.016', transform=ax.transAxes,
            facecolor='#F7F9FA', edgecolor=colour, linewidth=1.1))
        ax.text(.04, .84, heading, fontsize=14, color=colour, va='top',
                fontfamily='DejaVu Sans', transform=ax.transAxes)
        for y, line in zip([.56, .36, .16], lines):
            ax.text(.04, y, line, fontsize=10.7, color=ink, va='top',
                    fontfamily='DejaVu Sans', transform=ax.transAxes)
    return fig
