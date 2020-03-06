#  A Library of Multi-Object Tracking in Python and Pytorch

## Installation

environments: python 3.6, opencv 4.1.1

```bash
git clone https://github.com/nightmaredimple/libmot --recursive
cd libmot/
pip install -r requirements.txt
```

The details can be seen from my [blogs](https://huangpiao.tech/) or [zhihu](https://www.zhihu.com/people/huang-piao-72/posts).

## Feature Lists

|          Block          |               Method                |       Reference        | Complete |
| :---------------------: | :---------------------------------: | :--------------------: | :------: |
|                         |           IOU Assignment            |   iou-tracker&V-IOU    |    ✓     |
|                         |          Linear Assignment          |           -            |    ✓     |
|  **Data Association**   |             MinCostFlow             |          MCF           |    ✓     |
|                         |      Other End-to-End Network       |     FAMNet&DeepMOT     |    ☐     |
|                         |               GNN&GCN               |        MPNTrack        |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|                         |            Kalman Filter            |     Sort&DeepSort      |    ✓     |
|       **Motion**        |                 ECC                 |       Tracktor++       |    ☐     |
|                         |          Epipolar Geometry          |          TNT           |    ☐     |
|                         |            Optical Flow             |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|                         |                Re-ID                |           -            |    ☐     |
|     **Appearance**      |           Feature Fusion            |           -            |    ☐     |
|                         |          Feature Selection          |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|      **Detection**      |          Faster RCNN + FPN          |       Tracktor++       |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|                         |                 KCF                 |         KCF&CN         |    ☐     |
|         **SOT**         |          SiamRPN&SiamMask           |    SiamRPN&SiamMask    |    ☐     |
|                         |                DIMP                 |          DIMP          |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|                         |             DataLoader              |           -            |    ✓     |
|       **Tricks**        |          Spatial Blocking           |           -            |    ✓     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|                         |             Evaluation              |           -            |    ✓     |
|       **Others**        |        Tracking Visualiztion        |           -            |    ✓     |
|                         |        Feature Visualiztion         |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|      **Tracktor**       |             MIFT(ours)              |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|      **Detector**       |             MIFD(ours)              |           -            |    ☐     |



## Motion Model

```python
python scripts/test_kalman_tracker.py
```

 <div align="center">
  <img src="figures/kalman_tracker.png"  />
 </div>

## Data Association

 <div align="center">
  <img src="figures/linear_assignment.png"  />
 </div>

## Tracktor

Our proposed MIFT and MIFD will be released upon the acceptance on  ECCV20'

In MOT Challenge, the MIFT tracktor is named as ISE-MOT, the MIFD detector is named as ISE-MOTDet.

|  Method  | DataSets | MOTA↑ | IDF1↑ |  MT↑  |  ML↓  |  FP↓  |  FN↓   | ID Sw.↓ | Frag↓ | Hz↑  |
| :------: | :------: | :---: | :---: | :---: | :---: | :---: | :----: | :-----: | :---: | :--: |
|          |  MOT15   | 46.7  | 51.6  | 29.4% | 25.7% | 11003 | 20839  |   878   | 1265  | 6.7  |
| **MIFT** |  MOT16   | 60.1  | 56.9  | 26.1% | 29.1% | 6964  | 65044  |   739   |  951  | 6.9  |
|          |  MOT17   | 60.1  | 56.4  | 28.5% | 28.1% | 23168 | 199483 |  2556   | 3182  | 7.2  |

|  Method  | DataSets | AP↑  | MODA↑ | FAF↓ | Precision↑ | Recall↑ |
| :------: | :------: | :--: | :---: | :--: | :--------: | :-----: |
| **MIFD** | MOT17Det | 0.88 | 67.4  | 4.9  |    78.6    |  92.6   |

To be continued..