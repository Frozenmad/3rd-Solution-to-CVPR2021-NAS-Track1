# CVPR2021 NAS Track1 第三名技术方案：如何提升超网络一致性？

本项目由[AI Studio对应项目](https://aistudio.baidu.com/aistudio/projectdetail/2103110)迁移而来，请访问AI Studio获取对应项目运行方式。具体技术细节可参考[notebook说明](readme.ipynb)。您可以通过如下方式快捷地运行本项目：

## 数据集获取

首先下载[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/76994)，并放到如下文件夹下：
```
- data
    - cifar-100-python.tar.gz
    - Track1_final_archs.json
```

## 训练

通过如下命令来训练
```cmd
python -m supernet.scripts.train --n 18 --kd --sandwich
```

## 测试

通过如下命令来测试
```
python -m supernet.scripts.evaluate --path path/to/saved/model --output path/to/output
```

## 引用

如果您发现相关代码和项目对您有用，请引用我们的[文章](https://www.cvpr21-nas.com/resources/upload/0894f613d1c1/1624351688674/CVPR_Meta_Learners_track1_3.pdf)：
```
@inproceedings{guan2021oneshot,
    title={One-Shot Neural Channel Search: What Works and What’s Next},
    author={Chaoyu Guan and Yijian Qin and Zhikun Wei and Zeyang Zhang and Zizhao Zhang and Xin Wang and Wenwu Zhu},
    booktitle={CVPR 2021 Workshop on Neural Architecture Search: 1st lightweight NAS challenge and moving beyond},
    year={2021}
}
```
