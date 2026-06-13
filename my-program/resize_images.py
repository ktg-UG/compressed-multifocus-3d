#!/usr/bin/env python3
"""
画像リサイズツール
指定したフォルダ内のすべての画像を128x128にリサイズします。

python ./my-program/resize_images.py   -i ./RealDatas/   -o ./snapshot_Input/realData/ink/

"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def resize_images(input_dir, output_dir, target_size=(128, 128), interpolation='area'):
    """
    指定したフォルダ内のすべての画像を指定サイズにリサイズ
    
    Args:
        input_dir (str): 入力ディレクトリ
        output_dir (str): 出力ディレクトリ
        target_size (tuple): リサイズ後のサイズ (height, width)
        interpolation (str): 補間方法 ('area', 'linear', 'cubic', 'nearest')
    """
    # 補間方法の辞書
    interp_methods = {
        'area': cv2.INTER_AREA,      # ダウンサンプリングに最適
        'linear': cv2.INTER_LINEAR,  # 線形補間
        'cubic': cv2.INTER_CUBIC,    # 3次補間（高品質だが遅い）
        'nearest': cv2.INTER_NEAREST # 最近傍補間（高速だが低品質）
    }
    
    interp = interp_methods.get(interpolation.lower(), cv2.INTER_AREA)
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # サポートする画像形式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # 入力ディレクトリ内の全画像ファイルを取得
    input_path = Path(input_dir)
    image_files = [f for f in input_path.rglob('*') 
                   if f.suffix.lower() in image_extensions and f.is_file()]
    
    if not image_files:
        print(f"警告: {input_dir} に画像ファイルが見つかりませんでした")
        return
    
    print(f"見つかった画像: {len(image_files)}枚")
    print(f"リサイズサイズ: {target_size[0]}x{target_size[1]}")
    print(f"補間方法: {interpolation}")
    print(f"出力先: {output_dir}")
    print()
    
    # 統計情報
    success_count = 0
    error_count = 0
    
    # 各画像をリサイズ
    for img_file in tqdm(image_files, desc="リサイズ中"):
        try:
            # 相対パスを取得（サブディレクトリ構造を保持）
            rel_path = img_file.relative_to(input_path)
            output_file = Path(output_dir) / rel_path
            
            # 出力ディレクトリを作成
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 画像を読み込み
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"エラー: 読み込み失敗 - {img_file}")
                error_count += 1
                continue
            
            original_shape = img.shape
            
            # リサイズ
            # cv2.resize は (width, height) の順序
            resized = cv2.resize(img, (target_size[1], target_size[0]), 
                               interpolation=interp)
            
            # 保存
            cv2.imwrite(str(output_file), resized)
            success_count += 1
            
        except Exception as e:
            print(f"エラー: {img_file} - {e}")
            error_count += 1
    
    # 結果表示
    print()
    print("=" * 50)
    print(f"✅ 成功: {success_count}枚")
    if error_count > 0:
        print(f"❌ エラー: {error_count}枚")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='指定したフォルダ内のすべての画像を128x128にリサイズ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方
  python resize_images.py -i ./input_folder -o ./output_folder
  
  # サイズを指定
  python resize_images.py -i ./input_folder -o ./output_folder --size 256 256
  
  # 補間方法を指定
  python resize_images.py -i ./input_folder -o ./output_folder --interp cubic
  
  # snapshot画像をリサイズ
  python resize_images.py -i ../outputs/snapshot/realData/ink -o ../outputs/snapshot/realData/ink_128
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='入力ディレクトリパス')
    parser.add_argument('-o', '--output', required=True,
                       help='出力ディレクトリパス')
    parser.add_argument('--size', type=int, nargs=2, default=[128, 128],
                       metavar=('HEIGHT', 'WIDTH'),
                       help='リサイズ後のサイズ (デフォルト: 128 128)')
    parser.add_argument('--interp', choices=['area', 'linear', 'cubic', 'nearest'],
                       default='area',
                       help='補間方法 (デフォルト: area)')
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(args.input):
        print(f"エラー: 入力ディレクトリが存在しません: {args.input}")
        return
    
    # リサイズ実行
    resize_images(
        input_dir=args.input,
        output_dir=args.output,
        target_size=(args.size[0], args.size[1]),
        interpolation=args.interp
    )


if __name__ == '__main__':
    main()
