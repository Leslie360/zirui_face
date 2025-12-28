import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def draw_network():
    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Style constants
    cnn_color = '#E1F5FE' # Light Blue
    cnn_edge = '#0277BD'  # Dark Blue
    fc_color = '#FFF3E0'  # Light Orange
    fc_edge = '#EF6C00'   # Dark Orange
    text_color = '#333333'
    arrow_props = dict(facecolor='#666666', edgecolor='none', alpha=0.8, width=0.5, headwidth=4)

    # Helper function to draw a 3D-like box (Feature Map)
    def draw_box(x, y, h, w, d, label, sub_label, channels):
        # Front face
        rect = patches.Rectangle((x, y - h/2), w, h, linewidth=1.5, edgecolor=cnn_edge, facecolor=cnn_color, alpha=0.9)
        ax.add_patch(rect)
        
        # Side face (Depth)
        depth = d * 0.3 # Scale depth visual
        poly_side = np.array([
            [x + w, y - h/2],
            [x + w + depth, y - h/2 + depth],
            [x + w + depth, y + h/2 + depth],
            [x + w, y + h/2]
        ])
        ax.add_patch(patches.Polygon(poly_side, closed=True, linewidth=1, edgecolor=cnn_edge, facecolor='#B3E5FC'))
        
        # Top face
        poly_top = np.array([
            [x, y + h/2],
            [x + depth, y + h/2 + depth],
            [x + w + depth, y + h/2 + depth],
            [x + w, y + h/2]
        ])
        ax.add_patch(patches.Polygon(poly_top, closed=True, linewidth=1, edgecolor=cnn_edge, facecolor='#81D4FA'))
        
        # Labels
        ax.text(x + w/2, y - h/2 - 0.5, label, ha='center', va='top', fontsize=10, fontweight='bold', color=text_color)
        ax.text(x + w/2, y - h/2 - 1.0, sub_label, ha='center', va='top', fontsize=8, color='#555555')
        ax.text(x + w/2, y + h/2 + depth + 0.2, f"{channels} Ch", ha='center', va='bottom', fontsize=8, color='#0277BD')

        return x + w + depth, y

    # Helper function to draw FC layer (Circles)
    def draw_fc_layer(x, y, num_nodes, height, label, sub_label, is_output=False):
        # Draw representative nodes
        node_radius = 0.25
        spacing = height / (min(num_nodes, 6) + 1)
        
        positions = []
        display_nodes = 6 if num_nodes > 10 else num_nodes
        
        start_y = y + (display_nodes - 1) * spacing / 2
        
        for i in range(display_nodes):
            cy = start_y - i * spacing
            circle = patches.Circle((x, cy), node_radius, linewidth=1.5, edgecolor=fc_edge, facecolor=fc_color)
            ax.add_patch(circle)
            positions.append((x, cy))
            
            # Add ellipsis if simplified
            if i == display_nodes // 2 and num_nodes > 10:
                ax.text(x, cy - spacing/2, '...', ha='center', va='center', fontsize=12, fontweight='bold', color=fc_edge)

        # Labels
        ax.text(x, y - height/2 - 0.5, label, ha='center', va='top', fontsize=10, fontweight='bold', color=text_color)
        ax.text(x, y - height/2 - 1.0, sub_label, ha='center', va='top', fontsize=8, color='#555555')
        
        return positions

    # --- Draw Architecture ---

    # 1. Input
    cur_x = 1.0
    center_y = 6.0
    _, _ = draw_box(cur_x, center_y, 3.2, 0.5, 1.0, "Input", "32x32 Image", 3)
    
    # Arrow
    ax.annotate("", xy=(3.5, center_y), xytext=(2.8, center_y), arrowprops=arrow_props)
    
    # 2. Conv1
    cur_x = 3.5
    _, _ = draw_box(cur_x, center_y, 3.2, 0.8, 1.5, "Conv1", "Conv+BN+ReLU", 64)
    
    # Arrow
    ax.annotate("", xy=(6.0, center_y), xytext=(5.3, center_y), arrowprops=arrow_props)

    # 3. Layer 1 (ResBlock)
    cur_x = 6.0
    _, _ = draw_box(cur_x, center_y, 3.2, 1.2, 1.5, "Layer 1", "2x ResBlock\n(Stride 1)", 64)

    # Arrow (Downsample)
    ax.annotate("", xy=(8.5, center_y), xytext=(7.8, center_y), arrowprops=arrow_props)
    
    # 4. Layer 2 (ResBlock)
    cur_x = 8.5
    _, _ = draw_box(cur_x, center_y, 2.0, 1.2, 2.0, "Layer 2", "2x ResBlock\n(Stride 2)", 128)

    # Arrow (Downsample)
    ax.annotate("", xy=(11.0, center_y), xytext=(10.5, center_y), arrowprops=arrow_props)

    # 5. Layer 3 (ResBlock)
    cur_x = 11.0
    end_cnn_x, _ = draw_box(cur_x, center_y, 1.2, 1.2, 2.5, "Layer 3", "2x ResBlock\n(Stride 2)", 256)

    # Flatten Transition
    ax.text(14.0, center_y + 1.5, "Flatten", ha='center', va='bottom', fontsize=9, fontstyle='italic')
    ax.annotate("", xy=(15.0, center_y), xytext=(end_cnn_x + 0.2, center_y), arrowprops=arrow_props)
    
    # 6. FC1 (Classifier)
    cur_x = 15.0
    fc1_pos = draw_fc_layer(cur_x, center_y, 1024, 5.0, "FC Layer 1", "Linear+ReLU\n+Dropout", is_output=False)
    
    # Projection Head Branch (Optional but good for accuracy)
    # Drawing a small branch up for Projection Head
    proj_x = 15.0
    proj_y = 9.5
    # ax.annotate("", xy=(proj_x, proj_y), xytext=(14.0, center_y), arrowprops=dict(facecolor='#999999', edgecolor='none', alpha=0.5, width=0.3, headwidth=3, linestyle='--'))
    # draw_fc_layer(proj_x, proj_y, 128, 2.0, "Proj Head", "Contrastive\nOutput")
    
    # 7. Output Layer
    cur_x = 19.0
    out_pos = draw_fc_layer(cur_x, center_y, 10, 5.0, "Output", "Linear\n(Logits)", is_output=True)

    # Draw weights between FC layers
    # Just draw a few lines to imply full connection
    for p1 in fc1_pos:
        for p2 in out_pos:
            # Draw faint lines
            ax.plot([p1[0] + 0.25, p2[0] - 0.25], [p1[1], p2[1]], color='#CCCCCC', linewidth=0.5, alpha=0.3, zorder=0)

    # Title and Legend
    plt.suptitle("MemristorCNN_Strong Architecture Diagram", fontsize=16, fontweight='bold', y=0.95)
    
    # Add annotations for blocks
    # SE Block note
    ax.text(9.5, 9.0, "Note: Each ResBlock contains\nan SE (Squeeze-and-Excitation) Module", 
            ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#FBC02D", alpha=0.9))

    # Save
    plt.tight_layout()
    plt.savefig('network_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('network_diagram.svg', format='svg', bbox_inches='tight')
    print("Diagram generated: network_diagram.png, network_diagram.svg")

if __name__ == "__main__":
    try:
        draw_network()
    except Exception as e:
        import traceback
        traceback.print_exc()
