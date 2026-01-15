# gait_data_analysis の説明

## data_logger.py
10mを歩行して，以下のグラフのcsvデータとそのプロットを行う．

<video src="png/video.mp4" width="600" controls></video>

| 左足関節トルク | 右足関節トルク |
| :---: | :---: |
| <img src="png/Left_Torques.png" width="300"> | <img src="png/Right_Torques.png" width="300"> |

| 左足関節角速度 | 右足関節角速度 |
| :---: | :---: |
| <img src="png/Left_Velocity.png" width="300"> | <img src="png/Right_Velocity.png" width="300"> |

| 胴体の姿勢偏差 | 重心位置と足上げ高さ |
| :---: | :---: |
| <img src="png/stability_analysis.png" width="300"> | <img src="png/trajectory_side_view.png" width="300"> |

| 歩幅 | 2次元足跡 |
| :---: | :---: |
| <img src="png/stride_length_analysis.png" width="300"> | <img src="png/footprint_2d.png" width="300"> |

| 速度追従 | metrics_report |
| :---: | :---: |
| <img src="png/velocity_tracking.png" width="300"> | <img src="png/metrics_report.png" width="300"> |

## run_eval_test.sh
data_logger.pyを自動で実行するためのスクリプト．ただし，報酬項のタイトルを変更する必要がある．

## copy_data.sh
Docker内の上記グラフデータをポストへコピーする．