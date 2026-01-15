#!/bin/bash

# ------------------------------------------------------------------
# 設定項目
# ------------------------------------------------------------------

# Dockerコンテナ名
CONTAINER_NAME="isaac-lab-base"

# Docker内のデータ保存ルートディレクトリ (末尾のスラッシュなし)
DOCKER_SOURCE_ROOT="/workspace/isaaclab/ML3_g1_flat"

# 【変更点】ホスト側の保存先ルート
HOST_DEST_ROOT="/home/harukiogawa/g1_flat"

# コピー対象のファイルリスト
FILES_TO_COPY=(
    "data_forces.csv"
    "data_joints_all.csv"
    "footprint_2d.png"
    "metrics_report.txt"
    "trajectory_side_view.png"
    "velocity_tracking.png"
)

# 対象のディレクトリリスト
TARGET_DIRS=(
    "2025-12-22_12-04-51"
    "2025-12-22_12-35-53"
    "2025-12-22_13-06-56"
    "2025-12-22_13-38-10"
    "2025-12-25_12-39-40"
    "2025-12-25_13-10-42"
    "2025-12-25_13-41-47"
    "2025-12-25_14-12-58"
    "2025-12-25_14-43-53"
)

# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

echo "=== データコピー処理を開始します ==="
echo "保存先ルート: ${HOST_DEST_ROOT}"
echo "コンテナ名: ${CONTAINER_NAME}"
echo "対象ディレクトリ数: ${#TARGET_DIRS[@]}"

# 保存先ルートディレクトリ自体がなければ作成
if [ ! -d "$HOST_DEST_ROOT" ]; then
    echo "保存先ルートディレクトリを作成します: ${HOST_DEST_ROOT}"
    mkdir -p "$HOST_DEST_ROOT"
fi

for dir_name in "${TARGET_DIRS[@]}"
do
    # ディレクトリ名から末尾のスラッシュを除去
    clean_dir_name=$(basename "${dir_name}")
    
    # ホスト側の保存先パス (例: /home/harukiogawa/g1_flat/2025-12-22_12-04-51)
    LOCAL_DIR="${HOST_DEST_ROOT}/${clean_dir_name}"
    
    echo "------------------------------------------------------------"
    echo "処理中のディレクトリ: ${clean_dir_name}"

    # ホスト側に日時ディレクトリを作成
    if [ ! -d "$LOCAL_DIR" ]; then
        echo "フォルダ作成: ${LOCAL_DIR}"
        mkdir -p "$LOCAL_DIR"
    fi

    # 各ファイルをコピー
    for file_name in "${FILES_TO_COPY[@]}"
    do
        # Docker内のファイルパス
        SRC_PATH="${CONTAINER_NAME}:${DOCKER_SOURCE_ROOT}/${clean_dir_name}/${file_name}"
        
        # コピー実行
        docker cp "${SRC_PATH}" "${LOCAL_DIR}/"

        if [ $? -eq 0 ]; then
            echo "  [OK] Copied: ${file_name}"
        else
            echo "  [ERROR] Failed to copy: ${file_name}"
        fi
    done

done

echo "============================================================"
echo "すべての処理が完了しました。"
echo "確認コマンド: ls -R ${HOST_DEST_ROOT} | head -n 20"