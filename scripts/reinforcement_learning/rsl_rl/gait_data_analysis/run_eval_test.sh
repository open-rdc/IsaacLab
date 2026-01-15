#!/bin/bash

# ------------------------------------------------------------------
# 設定項目
# ------------------------------------------------------------------

# ログが保存されている親ディレクトリ
LOG_ROOT="/workspace/isaaclab/logs/rsl_rl/g1_flat"
#LOG_ROOT="/workspace/isaaclab/ML3_g1_flat_slide"

# 実行するPythonスクリプトのパス
PYTHON_SCRIPT="./scripts/reinforcement_learning/rsl_rl/data_logger_v2_x.py"

# タスク名
TASK_NAME="Isaac-Velocity-Flat-G1-v0"

# 使用するチェックポイントのモデルファイル名
MODEL_FILE="model_1499.pt"

TARGET_DIRS=(
    "2025-11-19_21-50-04"
    "2025-12-29_00-46-57"
    "2025-12-29_01-17-59"
    "2025-12-29_02-19-45"
    "2025-12-29_03-21-30"
    "2025-12-29_04-23-00"
    "2025-12-29_05-24-35"
)

# 報酬項はdata_logger.py内で設定している

WEIGHTS=(
    "0.0"
    "0.1"
    "0.2"
    "0.4"
    "0.6"
    "0.8"
    "1.0"
)


# ------------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------------

echo "=== バッチ処理を開始します ==="
echo "対象ディレクトリ数: ${#TARGET_DIRS[@]}"

# 配列のインデックスを使ってループする (${!TARGET_DIRS[@]} は 0 1 2... を返す)
for i in "${!TARGET_DIRS[@]}"
do
    dir_name="${TARGET_DIRS[$i]}"
    target_weight="${WEIGHTS[$i]}"

    # ディレクトリ名から末尾のスラッシュを除去
    clean_dir_name=$(basename "${dir_name}")
    
    # チェックポイントの絶対パスを生成
    CHECKPOINT_PATH="${LOG_ROOT}/${clean_dir_name}/${MODEL_FILE}"

    echo "------------------------------------------------------------"
    echo "処理中のディレクトリ: ${clean_dir_name}"
    echo "設定するWeight値  : ${target_weight}"
    
    # ファイルの存在確認
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "[WARNING] ファイルが見つかりません。スキップします: ${CHECKPOINT_PATH}"
        continue
    fi

    # 実行コマンドの構築
    # 【変更】--display_weight 引数を追加しました
    CMD="./isaaclab.sh -p ${PYTHON_SCRIPT} --task ${TASK_NAME} --num_envs 1 --checkpoint ${CHECKPOINT_PATH} --display_weight=${target_weight} --video --video_length 3000"
    
    echo "実行コマンド: ${CMD}"
    
    # コマンド実行
    $CMD
    
    # 終了ステータスの確認
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] ${clean_dir_name} (Weight: ${target_weight}) の処理が完了しました。"
    else
        echo "[ERROR] ${clean_dir_name} の実行中にエラーが発生しました。"
    fi
    
    sleep 2
done

echo "============================================================"
echo "すべての処理が完了しました。"