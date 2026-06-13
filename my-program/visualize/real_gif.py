import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import imageio.v2 as imageio
from pathlib import Path
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='results-gdrive/aneurism.npy', help='Input .npy file path')
parser.add_argument('-o', '--output', type=str, default='outputs/final_prezen_vis', help='Output directory')
parser.add_argument('--gif-only', action='store_true', help='Skip rendering, rebuild GIF from existing frames')
args = parser.parse_args()

# データ読み込み
print("Loading volume data...")
alphas = np.load(args.input, allow_pickle=True)
print(f"Shape: {alphas.shape}")
print(f"Min: {np.min(alphas):.4f}, Max: {np.max(alphas):.4f}")

# 物理サイズの設定 (マイクロメートル)
depth_resolution = 2.5      # 深さ方向: 1 μm/pixel
lateral_resolution = 1.0   # 横方向: 0.46 μm/pixel

# 実際の物理サイズを計算
depth_size = alphas.shape[0] * depth_resolution      # 11 μm
height_size = alphas.shape[1] * lateral_resolution   # 58.88 μm
width_size = alphas.shape[2] * lateral_resolution    # 56.88 μm

print(f"Physical size: {depth_size} x {height_size} x {width_size} μm")

# 3D可視化の準備
print("Creating 3D visualization...")
pio.templates.default = 'plotly_dark'
# pio.templates.default = 'plotly_white'

# 物理座標でグリッドを生成
X, Y, Z = np.mgrid[
    0:depth_size:depth_resolution,
    0:height_size:lateral_resolution,
    0:width_size:lateral_resolution
]



base_eye = dict(x=4.0, y=0.0, z=-5.0)
zoom_factor = 0.5
orbit_radius = np.hypot(base_eye["y"], base_eye["z"]) * zoom_factor
base_angle_rad = np.arctan2(base_eye["z"], base_eye["y"])
angles_deg = list(range(0, 360, 10))

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)
frame_duration = 5000.0 / len(angles_deg)  # ミリ秒単位

# 軸ラベルのスタイル設定
axis_label_font = dict(size=28, family='Arial Black', color='white')

scene_axes = dict(
    aspectmode='data',
    xaxis=dict(
        title='',
        showticklabels=False,
        backgroundcolor='rgba(0,0,0,0)',
        gridcolor='rgba(255,255,255,0.15)',
    ),
    yaxis=dict(
        title='',
        showticklabels=False,
        backgroundcolor='rgba(0,0,0,0)',
        gridcolor='rgba(255,255,255,0.15)',
        dtick=32,
        range=[0, 128],
    ),
    zaxis=dict(
        title='',
        showticklabels=False,
        backgroundcolor='rgba(0,0,0,0)',
        gridcolor='rgba(255,255,255,0.15)',
        dtick=32,
        range=[0, 128],
    ),
)

# 右上の軸インジケータ（ギズモ）設定
gizmo_scene = dict(
    aspectmode='cube',
    xaxis=dict(visible=False, range=[-2.0, 2.0]),
    yaxis=dict(visible=False, range=[-3.0, 2.0]),
    zaxis=dict(visible=False, range=[-2.0, 2.0]),
    bgcolor='rgba(0,0,0,0)',
)

# 軸矢印のデータ
def create_gizmo_traces(scene_name='scene2'):
    """3軸の矢印（線 + コーン）とラベルを生成"""
    axes = [
        # (direction, color, label, label_pos) - x=Depth, y=Height, z=Width
        ([-1, 0, 0], 'rgb(255, 80, 80)', 'D', [-2.0, -0.3, 0]),
        ([0, 1, 0], 'rgb(80, 255, 80)', 'H', [0, 1.35, 0]),
        ([0, 0, 1], 'rgb(80, 130, 255)', 'W', [0, 0, 1.35]),
    ]
    traces = []
    for (dx, dy, dz), color, label, (lx, ly, lz) in axes:
        # 軸の線
        traces.append(go.Scatter3d(
            x=[0, dx], y=[0, dy], z=[0, dz],
            mode='lines',
            line=dict(color=color, width=8),
            showlegend=False,
            scene=scene_name,
        ))
        # 矢印の先端（コーン）
        traces.append(go.Cone(
            x=[dx], y=[dy], z=[dz],
            u=[dx * 0.3], v=[dy * 0.3], w=[dz * 0.3],
            sizemode='absolute', sizeref=0.15,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            scene=scene_name,
        ))
        # ラベル
        traces.append(go.Scatter3d(
            x=[lx], y=[ly], z=[lz],
            mode='text',
            text=[label],
            textfont=dict(size=32, family='Arial Black', color=color),
            showlegend=False,
            scene=scene_name,
        ))
    return traces

# ビュー: orbit用（メイン + ギズモの2シーン）
vol_45 = go.Figure(
    data=[
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=alphas.flatten(),
            opacity=0.08,
            opacityscale=[[0.7019314169883728, 1], [1, 0]],
            surface_count=20,
            colorscale='turbo_r',
            isomin=0.7019314169883728,
            isomax=0.99,
            showscale=False,
            scene='scene',
        ),
    ] + create_gizmo_traces('scene2')
)

# サイズ情報の2Dアノテーション
size_annotations = [
    dict(
        text=f'<b>D: {alphas.shape[0]}</b>', x=0.02, y=0.98,
        xref='paper', yref='paper', showarrow=False,
        font=dict(size=24, family='Arial Black', color='rgb(255, 80, 80)'),
        xanchor='left', yanchor='top',
    ),
    dict(
        text=f'<b>H: {alphas.shape[1]}</b>', x=0.02, y=0.93,
        xref='paper', yref='paper', showarrow=False,
        font=dict(size=24, family='Arial Black', color='rgb(80, 255, 80)'),
        xanchor='left', yanchor='top',
    ),
    dict(
        text=f'<b>W: {alphas.shape[2]}</b>', x=0.02, y=0.88,
        xref='paper', yref='paper', showarrow=False,
        font=dict(size=24, family='Arial Black', color='rgb(80, 130, 255)'),
        xanchor='left', yanchor='top',
    ),
]

vol_45.update_layout(
    scene=dict(
        **scene_axes,
        domain=dict(x=[0, 1], y=[0, 1]),
    ),
    scene2=dict(
        **gizmo_scene,
        domain=dict(x=[0.6, 0.98], y=[0.78, 1.0]),
    ),
    annotations=size_annotations,
    height=800,
    margin=dict(l=0, r=0, t=30, b=0),
)

print("Rendering frames...")
frame_paths = []
if args.gif_only:
    frame_paths = sorted(output_dir.glob("frame_*.png"))
    print(f"Using {len(frame_paths)} existing frames")
else:
    for angle_deg in angles_deg:
        print(f"Rendering {angle_deg} deg...")
        start_time = time.time()
        angle_rad = base_angle_rad + np.deg2rad(angle_deg)
        # メインカメラ
        camera = dict(
            eye=dict(
                x=base_eye["x"] * zoom_factor,
                y=orbit_radius * np.cos(angle_rad),
                z=orbit_radius * np.sin(angle_rad)
            ),
            up=dict(x=1, y=0, z=0)
        )

        # ギズモ用カメラ（回転だけ同期、距離は固定）
        gizmo_dist = 1.0
        eye_vec = np.array([base_eye["x"] * zoom_factor, orbit_radius * np.cos(angle_rad), orbit_radius * np.sin(angle_rad)])
        eye_norm = eye_vec / np.linalg.norm(eye_vec) * gizmo_dist
        gizmo_camera = dict(
            eye=dict(x=float(eye_norm[0]), y=float(eye_norm[1]), z=float(eye_norm[2])),
            up=dict(x=1, y=0, z=0)
        )

        vol_45.update_layout(
            scene_camera=camera,
            scene2_camera=gizmo_camera,
        )

        frame_path = output_dir / f"frame_{angle_deg:03d}.png"
        vol_45.write_image(str(frame_path), width=1200, height=800, scale=1, engine="kaleido")
        frame_paths.append(frame_path)
        elapsed = time.time() - start_time
        print(f"Saved {frame_path} ({elapsed:.1f}s)")

print("Creating GIF...")
gif_path = output_dir / "orbit.gif"
with imageio.get_writer(gif_path, mode="I", duration=frame_duration, loop=0) as writer:
    for frame_path in frame_paths:
        frame = imageio.imread(frame_path)
        writer.append_data(frame)

print(f"GIF saved to: {gif_path}")