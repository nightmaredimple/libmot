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
|                         |           IOU Assignment            |   iou-tracker&V-IOU    |    ☐     |
|                         |          Linear Assignment          |           -            |    ✓     |
|  **Data Association**   |             MinCostFlow             |          MCF           |    ☐     |
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
|       **Tricks**        |          Spatial Blocking           |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|                         |             Evaluation              |           -            |    ✓     |
|       **Others**        |        Tracking Visualiztion        |           -            |    ✓     |
|                         |        Feature Visualiztion         |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|      **Tracktor**       |             MIFT(ours)              |           -            |    ☐     |
| ----------------------- | ----------------------------------- | ---------------------- |   ---    |
|      **Detector**       |             MIFD(ours)              |           -            |    ☐     |



## Motion Model

### 1.Kalman Filter

```python
python scripts/test_kalman_tracker.py
```

 <div align="center">
  <img src="figures/kalman_tracker.png"  />
 </div>


To be continued..