# 旷视AI智慧交通开源赛道第三名解决方案
**队伍名称：helloworld**<br/>
模型下载地址：https://pan.baidu.com/s/1EYDnp9OWsudAOWj-rlkgzQ <br/> 
提取码: s6rk
## 结果复现
进入trafficdet_gpu目录下：<br/>
&emsp;&emsp;将算法最终模型epoch_1.pkl拷贝至logs/faster_rcnn_res50_800size_trafficdet_demo_gpus2路径下<br/>
&emsp;&emsp;设置测试数据集软链接<br/>
&emsp;&emsp;执行eval.sh脚本
````
cd trafficdet_gpu
ln -s $data_root traffic5
sh eval.sh
````
## 训练说明
训练分为两个阶段，强数据增强策略的训练阶段（traffic_sign_v1）和弱数据增强策略微调（traffic_sign）<br/>
1.模型训练<br/>
进入trafficdet_gpu_v1目录下：<br/>
&emsp;&emsp;设置训练数据集软链接<br/>
&emsp;&emsp;在官网github或上面分享的模型链接下载resnet50预训练模型至weights文件夹<br/>
&emsp;&emsp;使用双卡训练算法<br/>
````
ln -s $data_root traffic5
sh frcn_demo.sh
````
2.模型微调<br/>
进入trafficdet_gpu目录下：<br/>
&emsp;&emsp;在训练脚本中指定上个阶段模型epoch_20.pkl为预加载模型<br/>
&emsp;&emsp;使用双卡训练，获得epoch_1.pkl之后停止训练，epoch_1.pkl作为最终推理模型。
````
sh frcn_demo.sh
````
## To do list
补充代码思路和进一步完善文档
