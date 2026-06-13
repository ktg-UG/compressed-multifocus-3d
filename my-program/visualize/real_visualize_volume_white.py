"""
実データ 正面+45度 (白背景バージョン)
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# データ読み込み
print("Loading volume data...")
alphas = np.load("results/MAUF/MaskedImagingModel_corrected/v3/realcell/0000/intermediate/volume_iter_03000.npy", allow_pickle=True)
print(f"Shape: {alphas.shape}")
print(f"Min: {np.min(alphas):.4f}, Max: {np.max(alphas):.4f}")

# 物理サイズの設定 (マイクロメートル)
depth_resolution = 1.0     # 深さ方向: 1 μm/pixel
lateral_resolution = 0.66   # 横方向: 0.46 μm/pixel

# 実際の物理サイズを計算
depth_size = alphas.shape[0] * depth_resolution      # 11 μm
height_size = alphas.shape[1] * lateral_resolution   # 58.88 μm
width_size = alphas.shape[2] * lateral_resolution    # 56.88 μm

print(f"Physical size: {depth_size} x {height_size} x {width_size} μm")

# 3D可視化の準備
print("Creating 3D visualization...")
pio.templates.default = 'plotly_white'

# 物理座標でグリッドを生成
X, Y, Z = np.mgrid[
    0:depth_size:depth_resolution,
    0:height_size:lateral_resolution,
    0:width_size:lateral_resolution
]

# 3Dシーンの軸スタイル(白背景用)
axis_style = dict(
    backgroundcolor='white',
    gridcolor='lightgray',
    zerolinecolor='lightgray',
    showbackground=True,
)

camera_front = dict(
    eye=dict(x=3.0, y=0.0, z=0.0),
    up=dict(x=0.0, y=-1.0, z=0.0)
)

vol_front = go.Figure(
    data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=alphas.flatten(),
        opacity=0.12,
        opacityscale=[[0.7019314169883728, 1], [1, 0]],
        surface_count=20,
        colorscale='turbo_r',
        isomin=0.7019314169883728,
        isomax=0.98,
    )
)

vol_front.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, **axis_style),
        yaxis=dict(title='Height', **axis_style),
        zaxis=dict(title='Width', **axis_style),
        aspectmode='data',
        camera=camera_front,
        bgcolor='white',
    ),
    paper_bgcolor='white',
    title="3D Volume Visualization (Front View)",
    height=800,
)

camera_45 = dict(
    eye=dict(x=2.0, y=-3.0, z=2.5),
    up=dict(x=1.0, y=0.0, z=0.0)
)

vol_45 = go.Figure(
    data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=alphas.flatten(),
        opacity=0.12,
        opacityscale=[[0.7019314169883728, 1], [1, 0]],
        surface_count=20,
        colorscale='turbo_r',
        isomin=0.7019314169883728,
        isomax=0.98,
    )
)

vol_45.update_layout(
    scene=dict(
        xaxis=dict(title='Depth(μm)', **axis_style),
        yaxis=dict(title='Height(μm)', **axis_style),
        zaxis=dict(title='Width(μm)', **axis_style),
        aspectmode='data',
        camera=camera_45,
        bgcolor='white',
    ),
    paper_bgcolor='white',
    title="3D Volume Visualization (45 deg View)",
    height=800,
)

print("Opening visualization in browser...")
vol_front.show()
vol_45.show()
