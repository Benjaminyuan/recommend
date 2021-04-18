# Neural Graph Collaborative Filtering

This is our Tensorflow implementation for the paper:

> Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering, [Paper in ACM DL](https://dl.acm.org/citation.cfm?doid=3331184.3331267) or [Paper in arXiv](https://arxiv.org/abs/1905.08108). In SIGIR'19, Paris, France, July 21-25, 2019.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

## Introduction

Neural Graph Collaborative Filtering (NGCF) is a new recommendation framework based on graph neural network, explicitly encoding the collaborative signal in the form of high-order connectivities in user-item bipartite graph by performing embedding propagation.

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{NGCF19,
  author    = {Xiang Wang and
               Xiangnan He and
               Meng Wang and
               Fuli Feng and
               Tat{-}Seng Chua},
  title     = {Neural Graph Collaborative Filtering},
  booktitle = {Proceedings of the 42nd International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2019, Paris,
               France, July 21-25, 2019.},
  pages     = {165--174},
  year      = {2019},
}
```

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

- tensorflow == 1.8.0
- numpy == 1.14.3
- scipy == 1.1.0
- sklearn == 0.19.1

## Example to Run the Codes

The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).

- Gowalla dataset

```
python NGCF.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

- Amazon-book dataset

```
python NGCF.py --dataset amazon-book --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 200 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

Some important arguments:

- `alg_type`

  - It specifies the type of graph convolutional layer.
  - Here we provide three options:
    - `ngcf` (by default), proposed in [Neural Graph Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir19-NGCF.pdf), SIGIR2019. Usage: `--alg_type ngcf`.
    - `gcn`, proposed in [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl), ICLR2018. Usage: `--alg_type gcn`.
    - `gcmc`, propsed in [Graph Convolutional Matrix Completion](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf), KDD2018. Usage: `--alg_type gcmc`.

- `adj_type`

  - It specifies the type of laplacian matrix where each entry defines the decay factor between two connected nodes.
  - Here we provide four options:
    - `ngcf` (by default), where each decay factor between two connected nodes is set as 1(out degree of the node), while each node is also assigned with 1 for self-connections. Usage: `--adj_type ngcf`.
    - `plain`, where each decay factor between two connected nodes is set as 1. No self-connections are considered. Usage: `--adj_type plain`.
    - `norm`, where each decay factor bewteen two connected nodes is set as 1/(out degree of the node + self-conncetion). Usage: `--adj_type norm`.
    - `gcmc`, where each decay factor between two connected nodes is set as 1/(out degree of the node). No self-connections are considered. Usage: `--adj_type gcmc`.

- `node_dropout`

  - It indicates the node dropout ratio, which randomly blocks a particular node and discard all its outgoing messages. Usage: `--node_dropout [0.1] --node_dropout_flag 1`
  - Note that the arguement `node_dropout_flag` also needs to be set as 1, since the node dropout could lead to higher computational cost compared to message dropout.

- `mess_dropout`
  - It indicates the message dropout ratio, which randomly drops out the outgoing messages. Usage `--mess_dropout [0.1,0.1,0.1]`.

## Dataset

We provide two processed datasets: Gowalla and Amazon-book.

- `train.txt`

  - Train file.
  - Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

- `test.txt`
  - Test file (positive instances).
  - Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  - Note that here we treat all unobserved interactions as the negative instances when reporting performance.
- `user_list.txt`
  - User file.
  - Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
- `item_list.txt`
  - Item file.
  - Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.

## Acknowledgement

This research is supported by the National Research Foundation, Singapore under its International Research Centres in Singapore Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

## 牛顿冷却公式

- [ruanyifeng](http://www.ruanyifeng.com/blog/2012/03/ranking_algorithm_newton_s_law_of_cooling.html)

## foursquare:

### command

    ```shell
    python3 main.py --data_path ./data/ --dataset square --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
    ```

### out

```shell
python3 main.py --data_path ./data/ --dataset square --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
n_users=4164, n_items=121142
n_interactions=484906
n_train=388672, n_test=96234, sparsity=0.00096
start----

already load adj matrix (125306, 125306) 0.07703542709350586
cuda:6
Traceback (most recent call last):
  File "main.py", line 131, in <module>
    train()
  File "main.py", line 25, in train
    norm_adj, args).to(args.device)
  File "/home/mist/recommend/model/NGCF.py", line 35, in __init__
    self.norm_adj).to(self.device)
RuntimeError: CUDA error: invalid device ordinal
Makefile:2: recipe for target 'square' failed
make: *** [square] Error 1

# mist @ MistGPU-193 in ~/recommend [10:20:31] C:2
$ make square
python3 main.py --data_path ./data/ --dataset square --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
n_users=4164, n_items=121142
n_interactions=484906
n_train=388672, n_test=96234, sparsity=0.00096
start----

already load adj matrix (125306, 125306) 0.08637571334838867
cuda:0
Epoch 0 [48.8s]: train==[218.93729=218.92015 + 0.01717]
Epoch 1 [46.4s]: train==[165.73357=165.71637 + 0.01721]
Epoch 2 [46.2s]: train==[137.89859=137.88132 + 0.01726]
Epoch 3 [46.4s]: train==[121.74956=121.73222 + 0.01732]
Epoch 4 [46.1s]: train==[110.51730=110.49994 + 0.01737]
Epoch 5 [46.4s]: train==[103.02088=103.00347 + 0.01741]
Epoch 6 [46.3s]: train==[96.68542=96.66799 + 0.01746]
Epoch 7 [46.3s]: train==[90.75254=90.73502 + 0.01750]
Epoch 8 [46.3s]: train==[84.98811=84.97054 + 0.01754]
Epoch 9 [46.2s + 40.7s]: train==[79.69659=79.67902 + 0.01759], recall=[0.00721, 0.01741], precision=[0.00347, 0.00195], hit=[0.06124, 0.14385], ndcg=[0.00642, 0.00977]
save the weights in path:  model/9.pkl
Epoch 10 [46.5s]: train==[75.99135=75.97376 + 0.01763]
Epoch 11 [46.4s]: train==[72.15642=72.13873 + 0.01768]
Epoch 12 [46.3s]: train==[69.23402=69.21631 + 0.01772]
Epoch 13 [46.4s]: train==[66.22005=66.20228 + 0.01776]
Epoch 14 [46.4s]: train==[63.39563=63.37778 + 0.01781]
Epoch 15 [46.2s]: train==[61.21404=61.19619 + 0.01785]
Epoch 16 [46.3s]: train==[59.29067=59.27277 + 0.01789]
Epoch 17 [46.2s]: train==[57.43890=57.42096 + 0.01793]
Epoch 18 [46.2s]: train==[56.50405=56.48610 + 0.01797]
Epoch 19 [46.1s + 40.9s]: train==[55.36177=55.34379 + 0.01802], recall=[0.00704, 0.01829], precision=[0.00346, 0.00222], hit=[0.06268, 0.16402], ndcg=[0.00662, 0.01055]
Epoch 20 [46.0s]: train==[54.21971=54.20162 + 0.01807]
Epoch 21 [46.3s]: train==[53.24991=53.23174 + 0.01811]
Epoch 22 [46.4s]: train==[52.46557=52.44740 + 0.01817]
Epoch 23 [46.3s]: train==[51.58000=51.56179 + 0.01822]
Epoch 24 [46.3s]: train==[50.76056=50.74227 + 0.01828]
Epoch 25 [46.3s]: train==[49.99860=49.98028 + 0.01833]
Epoch 26 [46.2s]: train==[49.16697=49.14859 + 0.01839]
Epoch 27 [46.2s]: train==[48.73285=48.71438 + 0.01846]
Epoch 28 [46.2s]: train==[47.95981=47.94130 + 0.01851]
Epoch 29 [46.1s + 41.1s]: train==[47.40500=47.38646 + 0.01858], recall=[0.00914, 0.02245], precision=[0.00456, 0.00265], hit=[0.08021, 0.19140], ndcg=[0.00846, 0.01298]
save the weights in path:  model/29.pkl
Epoch 30 [46.2s]: train==[46.78942=46.77078 + 0.01865]
Epoch 31 [46.4s]: train==[46.12660=46.10790 + 0.01871]
Epoch 32 [46.2s]: train==[45.46354=45.44478 + 0.01878]
Epoch 33 [46.5s]: train==[45.24724=45.22841 + 0.01884]
Epoch 34 [46.2s]: train==[44.45992=44.44102 + 0.01892]
Epoch 35 [46.1s]: train==[44.07212=44.05315 + 0.01899]
Epoch 36 [46.2s]: train==[43.57686=43.55783 + 0.01906]
Epoch 37 [46.4s]: train==[43.10481=43.08567 + 0.01913]
Epoch 38 [46.3s]: train==[42.74242=42.72318 + 0.01920]
Epoch 39 [46.1s + 41.2s]: train==[42.27221=42.25292 + 0.01928], recall=[0.01027, 0.02527], precision=[0.00519, 0.00322], hit=[0.09054, 0.21974], ndcg=[0.00999, 0.01541]
save the weights in path:  model/39.pkl
Epoch 40 [46.3s]: train==[41.75719=41.73783 + 0.01936]
Epoch 41 [46.2s]: train==[41.33381=41.31440 + 0.01943]
Epoch 42 [46.6s]: train==[40.86759=40.84808 + 0.01950]
Epoch 43 [46.7s]: train==[40.48503=40.46545 + 0.01958]
Epoch 44 [46.3s]: train==[40.15240=40.13276 + 0.01966]
Epoch 45 [46.2s]: train==[39.84218=39.82240 + 0.01975]
Epoch 46 [46.3s]: train==[39.18412=39.16428 + 0.01983]
Epoch 47 [46.3s]: train==[38.70929=38.68940 + 0.01991]
Epoch 48 [46.2s]: train==[38.67054=38.65057 + 0.01999]
Epoch 49 [46.5s + 40.9s]: train==[38.22820=38.20810 + 0.02008], recall=[0.01257, 0.02898], precision=[0.00576, 0.00345], hit=[0.09942, 0.23439], ndcg=[0.01180, 0.01754]
save the weights in path:  model/49.pkl
Epoch 50 [46.3s]: train==[37.94974=37.92957 + 0.02016]
Epoch 51 [46.1s]: train==[37.48588=37.46561 + 0.02026]
Epoch 52 [46.1s]: train==[37.18457=37.16421 + 0.02035]
Epoch 53 [46.1s]: train==[36.71272=36.69227 + 0.02044]
Epoch 54 [46.1s]: train==[36.45442=36.43388 + 0.02054]
Epoch 55 [46.0s]: train==[36.28472=36.26409 + 0.02062]
Epoch 56 [46.2s]: train==[35.94389=35.92318 + 0.02072]
Epoch 57 [46.2s]: train==[35.71137=35.69054 + 0.02082]
Epoch 58 [46.1s]: train==[35.40754=35.38666 + 0.02091]
Epoch 59 [46.1s + 41.7s]: train==[35.12762=35.10661 + 0.02102], recall=[0.01443, 0.03235], precision=[0.00680, 0.00380], hit=[0.11407, 0.25144], ndcg=[0.01435, 0.02038]
save the weights in path:  model/59.pkl
Epoch 60 [46.9s]: train==[34.86577=34.84467 + 0.02110]
Epoch 61 [46.5s]: train==[34.24694=34.22572 + 0.02122]
Epoch 62 [46.6s]: train==[34.08670=34.06538 + 0.02130]
Epoch 63 [46.7s]: train==[33.80118=33.77976 + 0.02141]
Epoch 64 [46.7s]: train==[33.60602=33.58451 + 0.02153]
Epoch 65 [46.5s]: train==[33.25595=33.23430 + 0.02163]
Epoch 66 [46.6s]: train==[33.04753=33.02582 + 0.02173]
Epoch 67 [46.6s]: train==[32.69853=32.67669 + 0.02183]
Epoch 68 [46.6s]: train==[32.60807=32.58613 + 0.02192]
Epoch 69 [46.4s + 41.4s]: train==[32.24156=32.21952 + 0.02204], recall=[0.01590, 0.03448], precision=[0.00748, 0.00416], hit=[0.12464, 0.27089], ndcg=[0.01624, 0.02276]
save the weights in path:  model/69.pkl
Epoch 70 [46.2s]: train==[32.05293=32.03080 + 0.02214]
Epoch 71 [46.2s]: train==[31.60447=31.58219 + 0.02226]
Epoch 72 [46.3s]: train==[31.56279=31.54042 + 0.02237]
Epoch 73 [46.2s]: train==[31.20318=31.18071 + 0.02247]
Epoch 74 [46.3s]: train==[30.88034=30.85777 + 0.02258]
Epoch 75 [46.6s]: train==[30.65719=30.63453 + 0.02267]
Epoch 76 [46.2s]: train==[30.42723=30.40438 + 0.02282]
Epoch 77 [46.2s]: train==[30.24585=30.22295 + 0.02292]
Epoch 78 [46.2s]: train==[30.00321=29.98019 + 0.02302]
Epoch 79 [46.4s + 41.7s]: train==[29.80709=29.78396 + 0.02314], recall=[0.01781, 0.03750], precision=[0.00863, 0.00469], hit=[0.14337, 0.29443], ndcg=[0.01845, 0.02540]
save the weights in path:  model/79.pkl
Epoch 80 [46.5s]: train==[29.62270=29.59945 + 0.02325]
Epoch 81 [46.4s]: train==[29.44235=29.41899 + 0.02336]
Epoch 82 [46.3s]: train==[29.24531=29.22186 + 0.02347]
Epoch 83 [46.3s]: train==[28.94855=28.92498 + 0.02357]
Epoch 84 [46.2s]: train==[28.71960=28.69592 + 0.02368]
Epoch 85 [46.2s]: train==[28.51118=28.48736 + 0.02380]
Epoch 86 [46.2s]: train==[28.18707=28.16316 + 0.02391]
Epoch 87 [46.2s]: train==[28.07853=28.05451 + 0.02404]
Epoch 88 [46.2s]: train==[27.91512=27.89096 + 0.02416]
Epoch 89 [46.5s + 41.4s]: train==[27.68422=27.65999 + 0.02424], recall=[0.01858, 0.03981], precision=[0.00964, 0.00501], hit=[0.15514, 0.31148], ndcg=[0.02013, 0.02731]
save the weights in path:  model/89.pkl
Epoch 90 [46.2s]: train==[27.55773=27.53334 + 0.02437]
Epoch 91 [46.1s]: train==[27.32598=27.30152 + 0.02448]
Epoch 92 [46.1s]: train==[27.18126=27.15670 + 0.02457]
Epoch 93 [46.0s]: train==[26.87772=26.85301 + 0.02472]
Epoch 94 [46.1s]: train==[26.77850=26.75366 + 0.02484]
Epoch 95 [46.0s]: train==[26.50789=26.48293 + 0.02495]
Epoch 96 [46.2s]: train==[26.31936=26.29429 + 0.02507]
Epoch 97 [46.2s]: train==[26.16242=26.13722 + 0.02519]
Epoch 98 [46.1s]: train==[26.00511=25.97981 + 0.02530]
Epoch 99 [46.2s + 41.4s]: train==[25.72784=25.70241 + 0.02541], recall=[0.01985, 0.04149], precision=[0.01051, 0.00553], hit=[0.16667, 0.32925], ndcg=[0.02168, 0.02919]
save the weights in path:  model/99.pkl
Epoch 100 [46.6s]: train==[25.53297=25.50744 + 0.02554]
Epoch 101 [46.5s]: train==[25.56954=25.54390 + 0.02564]
Epoch 102 [46.7s]: train==[25.36851=25.34274 + 0.02577]
Epoch 103 [46.7s]: train==[25.06974=25.04389 + 0.02585]
Epoch 104 [46.7s]: train==[24.94115=24.91514 + 0.02600]
Epoch 105 [46.5s]: train==[24.85537=24.82930 + 0.02608]
Epoch 106 [46.5s]: train==[24.51735=24.49115 + 0.02621]
Epoch 107 [46.4s]: train==[24.29054=24.26422 + 0.02633]
Epoch 108 [46.4s]: train==[24.21698=24.19052 + 0.02645]
Epoch 109 [46.3s + 41.6s]: train==[24.05305=24.02649 + 0.02656], recall=[0.02091, 0.04349], precision=[0.01215, 0.00609], hit=[0.18972, 0.35207], ndcg=[0.02436, 0.03184]
save the weights in path:  model/109.pkl
Epoch 110 [46.2s]: train==[23.90247=23.87577 + 0.02670]
Epoch 111 [46.2s]: train==[23.59604=23.56924 + 0.02679]
Epoch 112 [46.0s]: train==[23.44042=23.41351 + 0.02692]
Epoch 113 [46.0s]: train==[23.24445=23.21742 + 0.02703]
Epoch 114 [46.1s]: train==[23.14273=23.11560 + 0.02714]
Epoch 115 [46.0s]: train==[22.91564=22.88836 + 0.02726]
Epoch 116 [46.2s]: train==[22.83571=22.80836 + 0.02736]
Epoch 117 [46.4s]: train==[22.71340=22.68592 + 0.02748]
Epoch 118 [46.2s]: train==[22.34068=22.31311 + 0.02759]
Epoch 119 [46.5s + 41.3s]: train==[22.22345=22.19573 + 0.02773], recall=[0.02216, 0.04591], precision=[0.01309, 0.00657], hit=[0.20293, 0.36623], ndcg=[0.02589, 0.03362]
save the weights in path:  model/119.pkl
Epoch 120 [46.5s]: train==[22.15991=22.13209 + 0.02783]
Epoch 121 [46.5s]: train==[21.88876=21.86080 + 0.02796]
Epoch 122 [46.5s]: train==[21.80496=21.77688 + 0.02808]
Epoch 123 [46.4s]: train==[21.51588=21.48770 + 0.02818]
Epoch 124 [46.4s]: train==[21.46587=21.43757 + 0.02831]
Epoch 125 [46.4s]: train==[21.31156=21.28316 + 0.02840]
Epoch 126 [46.5s]: train==[21.16012=21.13160 + 0.02854]
Epoch 127 [46.4s]: train==[21.12538=21.09672 + 0.02865]
Epoch 128 [46.2s]: train==[20.83302=20.80426 + 0.02877]
Epoch 129 [46.2s + 41.2s]: train==[20.65832=20.62944 + 0.02888], recall=[0.02243, 0.04650], precision=[0.01389, 0.00677], hit=[0.21374, 0.37272], ndcg=[0.02719, 0.03477]
save the weights in path:  model/129.pkl
Epoch 130 [46.3s]: train==[20.48504=20.45605 + 0.02900]
Epoch 131 [46.2s]: train==[20.46926=20.44013 + 0.02912]
Epoch 132 [46.0s]: train==[20.22909=20.19987 + 0.02923]
Epoch 133 [45.9s]: train==[20.20747=20.17812 + 0.02936]
Epoch 134 [45.9s]: train==[20.00727=19.97777 + 0.02949]
Epoch 135 [46.0s]: train==[19.78835=19.75879 + 0.02956]
Epoch 136 [46.1s]: train==[19.76195=19.73227 + 0.02968]
Epoch 137 [46.5s]: train==[19.45703=19.42722 + 0.02981]
Epoch 138 [46.0s]: train==[19.40799=19.37807 + 0.02992]
Epoch 139 [46.0s + 41.4s]: train==[19.23959=19.20955 + 0.03005], recall=[0.02320, 0.04772], precision=[0.01475, 0.00711], hit=[0.22286, 0.38425], ndcg=[0.02853, 0.03615]
save the weights in path:  model/139.pkl
Epoch 140 [46.0s]: train==[18.90959=18.87942 + 0.03016]
Epoch 141 [45.8s]: train==[18.86731=18.83702 + 0.03029]
Epoch 142 [45.9s]: train==[18.75335=18.72297 + 0.03038]
Epoch 143 [45.8s]: train==[18.54844=18.51792 + 0.03052]
Epoch 144 [45.9s]: train==[18.43996=18.40934 + 0.03062]
Epoch 145 [45.7s]: train==[18.26748=18.23674 + 0.03074]
Epoch 146 [46.1s]: train==[18.15669=18.12586 + 0.03082]
Epoch 147 [45.8s]: train==[17.94067=17.90970 + 0.03099]
Epoch 148 [45.9s]: train==[17.70325=17.67216 + 0.03109]
Epoch 149 [46.4s + 41.8s]: train==[17.61972=17.58850 + 0.03121], recall=[0.02324, 0.04915], precision=[0.01517, 0.00743], hit=[0.23055, 0.39457], ndcg=[0.02942, 0.03732]
save the weights in path:  model/149.pkl
Epoch 150 [46.2s]: train==[17.50098=17.46962 + 0.03137]
Epoch 151 [46.1s]: train==[17.44488=17.41343 + 0.03145]
Epoch 152 [46.1s]: train==[17.29351=17.26195 + 0.03156]
Epoch 153 [46.1s]: train==[17.15605=17.12436 + 0.03169]
Epoch 154 [46.4s]: train==[17.02847=16.99667 + 0.03180]
Epoch 155 [46.1s]: train==[16.77007=16.73816 + 0.03191]
Epoch 156 [46.1s]: train==[16.73821=16.70616 + 0.03205]
Epoch 157 [46.1s]: train==[16.56502=16.53285 + 0.03218]
Epoch 158 [46.0s]: train==[16.44621=16.41393 + 0.03228]
Epoch 159 [46.0s + 41.4s]: train==[16.39114=16.35878 + 0.03236], recall=[0.02377, 0.04963], precision=[0.01563, 0.00769], hit=[0.23655, 0.40298], ndcg=[0.02945, 0.03746]
save the weights in path:  model/159.pkl
Epoch 160 [46.5s]: train==[16.23018=16.19770 + 0.03250]
Epoch 161 [46.5s]: train==[16.18639=16.15375 + 0.03264]
Epoch 162 [46.4s]: train==[15.88559=15.85283 + 0.03275]
Epoch 163 [46.4s]: train==[15.76901=15.73615 + 0.03286]
Epoch 164 [46.3s]: train==[15.70053=15.66756 + 0.03298]
Epoch 165 [46.4s]: train==[15.53940=15.50632 + 0.03308]
Epoch 166 [46.2s]: train==[15.44004=15.40681 + 0.03322]
Epoch 167 [46.4s]: train==[15.18317=15.14983 + 0.03334]
Epoch 168 [46.4s]: train==[15.28374=15.25030 + 0.03343]
Epoch 169 [46.3s + 42.1s]: train==[15.07255=15.03898 + 0.03357], recall=[0.02406, 0.05029], precision=[0.01607, 0.00790], hit=[0.24280, 0.40658], ndcg=[0.03013, 0.03818]
save the weights in path:  model/169.pkl
Epoch 170 [45.9s]: train==[14.91628=14.88263 + 0.03364]
Epoch 171 [46.1s]: train==[14.80496=14.77114 + 0.03383]
Epoch 172 [45.9s]: train==[14.65967=14.62575 + 0.03392]
Epoch 173 [46.0s]: train==[14.55051=14.51649 + 0.03402]
Epoch 174 [46.0s]: train==[14.43923=14.40509 + 0.03414]
Epoch 175 [45.9s]: train==[14.39663=14.36236 + 0.03426]
Epoch 176 [46.0s]: train==[14.22016=14.18574 + 0.03440]
Epoch 177 [45.9s]: train==[14.09423=14.05974 + 0.03450]
Epoch 178 [46.4s]: train==[14.08425=14.04967 + 0.03459]
Epoch 179 [46.4s + 41.7s]: train==[13.94449=13.90980 + 0.03469], recall=[0.02356, 0.05029], precision=[0.01604, 0.00797], hit=[0.24063, 0.40706], ndcg=[0.03022, 0.03836]
Epoch 180 [46.3s]: train==[13.77253=13.73768 + 0.03484]
Epoch 181 [46.1s]: train==[13.60159=13.56664 + 0.03495]
Epoch 182 [46.1s]: train==[13.37699=13.34189 + 0.03510]
Epoch 183 [46.1s]: train==[13.42372=13.38855 + 0.03518]
Epoch 184 [46.4s]: train==[13.23032=13.19500 + 0.03532]
Epoch 185 [46.2s]: train==[13.14602=13.11061 + 0.03540]
Epoch 186 [46.1s]: train==[13.00333=12.96779 + 0.03554]
Epoch 187 [46.2s]: train==[12.96277=12.92712 + 0.03565]
Epoch 188 [46.1s]: train==[12.92175=12.88597 + 0.03577]
Epoch 189 [46.2s + 41.4s]: train==[12.82214=12.78622 + 0.03591], recall=[0.02325, 0.05140], precision=[0.01640, 0.00820], hit=[0.24159, 0.41234], ndcg=[0.03030, 0.03873]
Epoch 190 [46.5s]: train==[12.58771=12.55167 + 0.03604]
Epoch 191 [46.4s]: train==[12.51253=12.47641 + 0.03613]
Epoch 192 [46.5s]: train==[12.59332=12.55710 + 0.03622]
Epoch 193 [46.6s]: train==[12.26774=12.23138 + 0.03636]
Epoch 194 [46.4s]: train==[12.26803=12.23156 + 0.03647]
Epoch 195 [46.4s]: train==[12.15384=12.11724 + 0.03660]
Epoch 196 [46.5s]: train==[12.00885=11.97213 + 0.03672]
Epoch 197 [46.4s]: train==[11.84597=11.80910 + 0.03686]
Epoch 198 [46.6s]: train==[11.78879=11.75182 + 0.03697]
Epoch 199 [46.9s + 41.8s]: train==[11.68396=11.64688 + 0.03709], recall=[0.02423, 0.05047], precision=[0.01691, 0.00820], hit=[0.24688, 0.41378], ndcg=[0.03086, 0.03867]
save the weights in path:  model/199.pkl
Epoch 200 [46.2s]: train==[11.54345=11.50624 + 0.03720]
Epoch 201 [46.4s]: train==[11.46339=11.42608 + 0.03730]
Epoch 202 [46.3s]: train==[11.32446=11.28704 + 0.03742]
Epoch 203 [46.4s]: train==[11.32402=11.28650 + 0.03752]
Epoch 204 [46.3s]: train==[11.18836=11.15072 + 0.03765]
Epoch 205 [46.4s]: train==[11.16062=11.12284 + 0.03778]
Epoch 206 [46.0s]: train==[11.09499=11.05710 + 0.03788]
Epoch 207 [46.2s]: train==[10.86568=10.82768 + 0.03800]
Epoch 208 [46.1s]: train==[10.73260=10.69454 + 0.03807]
$ Epoch 209 [46.3s + 47.1s]: train==[10.74080=10.70256 + 0.03824], recall=[0.02414, 0.05210], precision=[0.01726, 0.00833], hit=[0.24760, 0.41907], ndcg=[0.03121, 0.03932]
$ Epoch 210 [46.2s]: train==[10.62715=10.58882 + 0.03833]
Epoch 211 [45.9s]: train==[10.61723=10.57875 + 0.03849]
Epoch 212 [45.9s]: train==[10.46577=10.42716 + 0.03861]
Epoch 213 [45.9s]: train==[10.52848=10.48980 + 0.03869]
Epoch 214 [46.0s]: train==[10.37170=10.33291 + 0.03879]
Epoch 215 [45.9s]: train==[10.10033=10.06142 + 0.03892]
Epoch 216 [46.1s]: train==[10.05873=10.01970 + 0.03903]
Epoch 217 [45.9s]: train==[10.02548=9.98633 + 0.03915]
Epoch 218 [46.2s]: train==[9.88521=9.84595 + 0.03926]
Epoch 219 [46.0s + 41.4s]: train==[9.83074=9.79136 + 0.03938], recall=[0.02377, 0.05119], precision=[0.01709, 0.00838], hit=[0.24688, 0.42171], ndcg=[0.03098, 0.03910]
Epoch 220 [45.9s]: train==[9.77439=9.73485 + 0.03954]
Epoch 221 [45.9s]: train==[9.66933=9.62968 + 0.03964]
Epoch 222 [45.9s]: train==[9.64936=9.60966 + 0.03970]
Epoch 223 [45.9s]: train==[9.57887=9.53903 + 0.03983]
Epoch 224 [45.9s]: train==[9.50766=9.46771 + 0.03995]
Epoch 225 [45.8s]: train==[9.35317=9.31308 + 0.04010]
Epoch 226 [45.8s]: train==[9.27032=9.23011 + 0.04021]
Epoch 227 [45.9s]: train==[9.16897=9.12867 + 0.04029]
Epoch 228 [45.9s]: train==[9.20423=9.16379 + 0.04044]
Epoch 229 [45.9s + 41.5s]: train==[9.05846=9.01792 + 0.04054], recall=[0.02337, 0.05165], precision=[0.01680, 0.00838], hit=[0.24688, 0.41979], ndcg=[0.03067, 0.03907]
Epoch 230 [45.9s]: train==[8.96751=8.92686 + 0.04065]
Epoch 231 [45.9s]: train==[8.95142=8.91067 + 0.04074]
Epoch 232 [46.0s]: train==[8.80358=8.76270 + 0.04088]
Epoch 233 [45.9s]: train==[8.86552=8.82454 + 0.04098]
Epoch 234 [46.0s]: train==[8.64803=8.60693 + 0.04109]
Epoch 235 [45.9s]: train==[8.58582=8.54459 + 0.04123]
Epoch 236 [46.0s]: train==[8.59634=8.55497 + 0.04137]
Epoch 237 [46.2s]: train==[8.40050=8.35903 + 0.04147]
Epoch 238 [45.9s]: train==[8.43353=8.39194 + 0.04159]
Epoch 239 [45.9s + 41.5s]: train==[8.40572=8.36402 + 0.04169], recall=[0.02339, 0.05133], precision=[0.01715, 0.00838], hit=[0.24904, 0.41931], ndcg=[0.03064, 0.03888]
Epoch 240 [46.1s]: train==[8.25020=8.20841 + 0.04178]
Epoch 241 [45.9s]: train==[8.23570=8.19379 + 0.04190]
Epoch 242 [45.9s]: train==[8.04618=8.00417 + 0.04201]
Epoch 243 [45.9s]: train==[7.97726=7.93510 + 0.04216]
Epoch 244 [45.8s]: train==[7.97512=7.93288 + 0.04224]
Epoch 245 [46.0s]: train==[8.01296=7.97061 + 0.04235]
Epoch 246 [46.1s]: train==[7.86858=7.82612 + 0.04245]
Epoch 247 [45.9s]: train==[7.89088=7.84830 + 0.04258]
Epoch 248 [45.9s]: train==[7.69057=7.64786 + 0.04271]
Epoch 249 [46.1s + 41.4s]: train==[7.76424=7.72144 + 0.04280], recall=[0.02303, 0.05151], precision=[0.01695, 0.00840], hit=[0.25000, 0.41883], ndcg=[0.03053, 0.03886]
Early stopping is trigger at step: 5 log:0.023029916629476895
Best Iter=[19]@[12597.5]        recall=[0.02423 0.03260 0.03968 0.04571  0.05047], precision=[0.01691    0.01221 0.01035 0.00907 0.00820], hit=[0.24688   0.31436 0.35831 0.38857 0.41378], ndcg=[0.03086  0.03186 0.03443 0.03674 0.03867]

```

## gowalla

### out

```shell
nohup: ignoring input
python3 main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
n_users=29858, n_items=40981
n_interactions=1027370
n_train=810128, n_test=217242, sparsity=0.00084
cuda:0
already load adj matrix (70839, 70839) 0.14722633361816406
Epoch 0 [55.5s]: train==[407.78668=407.75085 + 0.03580]
Epoch 1 [55.6s]: train==[230.94626=230.91002 + 0.03620]
Epoch 2 [55.6s]: train==[191.46007=191.42360 + 0.03647]
Epoch 3 [58.8s]: train==[174.67708=174.64043 + 0.03666]
Epoch 4 [55.6s]: train==[161.98724=161.95038 + 0.03682]
Epoch 5 [55.7s]: train==[152.18739=152.15038 + 0.03695]
Epoch 6 [55.7s]: train==[143.92059=143.88351 + 0.03707]
Epoch 7 [55.7s]: train==[138.41190=138.37463 + 0.03719]
Epoch 8 [55.6s]: train==[134.18936=134.15202 + 0.03730]
Epoch 9 [55.7s + 194.1s]: train==[130.46216=130.42471 + 0.03742], recall=[0.03291, 0.11206], precision=[0.01082, 0.00730], hit=[0.16977, 0.40897], ndcg=[0.02342, 0.04704]
save the weights in path:  model/9.pkl
Epoch 10 [55.6s]: train==[127.04795=127.01029 + 0.03754]
Epoch 11 [55.5s]: train==[123.77515=123.73743 + 0.03767]
Epoch 12 [55.6s]: train==[120.13998=120.10211 + 0.03782]
Epoch 13 [55.6s]: train==[117.23463=117.19686 + 0.03799]
Epoch 14 [55.6s]: train==[115.53403=115.49593 + 0.03816]
Epoch 15 [55.6s]: train==[113.40416=113.36582 + 0.03834]
Epoch 16 [55.6s]: train==[111.41415=111.37560 + 0.03854]
Epoch 17 [55.6s]: train==[109.50169=109.46297 + 0.03875]
Epoch 18 [55.6s]: train==[107.85264=107.81374 + 0.03897]
Epoch 19 [55.6s + 192.2s]: train==[106.22054=106.18144 + 0.03919], recall=[0.06307, 0.17614], precision=[0.01926, 0.01101], hit=[0.28230, 0.54538], ndcg=[0.04754, 0.08129]
save the weights in path:  model/19.pkl
Epoch 20 [55.6s]: train==[104.78189=104.74252 + 0.03942]
Epoch 21 [55.5s]: train==[103.37466=103.33498 + 0.03968]
Epoch 22 [55.6s]: train==[101.78077=101.74089 + 0.03993]
Epoch 23 [55.6s]: train==[100.31059=100.27037 + 0.04019]
Epoch 24 [55.6s]: train==[98.38457=98.34409 + 0.04046]
Epoch 25 [55.6s]: train==[97.34707=97.30624 + 0.04075]
Epoch 26 [55.6s]: train==[95.70380=95.66286 + 0.04106]
Epoch 27 [55.6s]: train==[94.52977=94.48839 + 0.04135]
Epoch 28 [55.6s]: train==[93.04984=93.00813 + 0.04167]
Epoch 29 [55.6s + 192.8s]: train==[91.38438=91.34241 + 0.04198], recall=[0.09274, 0.22554], precision=[0.02794, 0.01399], hit=[0.37742, 0.63223], ndcg=[0.07451, 0.11407]
save the weights in path:  model/29.pkl
Epoch 30 [55.8s]: train==[90.67788=90.63557 + 0.04231]
Epoch 31 [55.6s]: train==[88.90814=88.86544 + 0.04266]
Epoch 32 [55.6s]: train==[87.85799=87.81501 + 0.04299]
Epoch 33 [55.7s]: train==[86.32362=86.28024 + 0.04337]
Epoch 34 [55.7s]: train==[85.10239=85.05861 + 0.04373]
Epoch 35 [55.6s]: train==[83.74090=83.69669 + 0.04411]
Epoch 36 [55.7s]: train==[82.51046=82.46593 + 0.04451]
Epoch 37 [55.7s]: train==[81.47638=81.43144 + 0.04487]
Epoch 38 [55.6s]: train==[80.19514=80.14987 + 0.04529]
Epoch 39 [55.8s + 194.3s]: train==[78.72272=78.67706 + 0.04571], recall=[0.11320, 0.25418], precision=[0.03427, 0.01590], hit=[0.43894, 0.68437], ndcg=[0.09480, 0.13663]
save the weights in path:  model/39.pkl
Epoch 40 [55.6s]: train==[77.64633=77.60011 + 0.04613]
Epoch 41 [55.4s]: train==[76.35444=76.30789 + 0.04656]
Epoch 42 [55.5s]: train==[75.16211=75.11519 + 0.04698]
Epoch 43 [55.6s]: train==[74.13493=74.08761 + 0.04740]
Epoch 44 [55.6s]: train==[72.86086=72.81303 + 0.04787]
Epoch 45 [55.6s]: train==[71.62057=71.57227 + 0.04831]
Epoch 46 [55.5s]: train==[70.70593=70.65720 + 0.04874]
Epoch 47 [55.6s]: train==[69.52028=69.47104 + 0.04923]
Epoch 48 [55.5s]: train==[68.77924=68.72955 + 0.04968]
Epoch 49 [55.6s + 195.0s]: train==[67.50789=67.45781 + 0.05012], recall=[0.12380, 0.27019], precision=[0.03800, 0.01700], hit=[0.47468, 0.71220], ndcg=[0.10610, 0.14926]
save the weights in path:  model/49.pkl
Epoch 50 [55.6s]: train==[66.34733=66.29690 + 0.05056]
Epoch 51 [55.5s]: train==[65.34820=65.29719 + 0.05104]
Epoch 52 [55.6s]: train==[64.34610=64.29459 + 0.05149]
Epoch 53 [55.6s]: train==[63.46456=63.41254 + 0.05197]
Epoch 54 [55.6s]: train==[62.31020=62.25779 + 0.05243]
Epoch 55 [55.6s]: train==[61.74017=61.68725 + 0.05289]
Epoch 56 [55.6s]: train==[60.39767=60.34425 + 0.05335]
Epoch 57 [55.6s]: train==[59.75967=59.70581 + 0.05381]
Epoch 58 [55.6s]: train==[58.58816=58.53393 + 0.05428]
Epoch 59 [55.6s + 193.8s]: train==[58.14252=58.08786 + 0.05474], recall=[0.12818, 0.27729], precision=[0.03964, 0.01753], hit=[0.48607, 0.72219], ndcg=[0.11124, 0.15502]
save the weights in path:  model/59.pkl
Epoch 60 [55.7s]: train==[57.24609=57.19093 + 0.05519]
Epoch 61 [55.7s]: train==[56.47783=56.42215 + 0.05564]
Epoch 62 [55.6s]: train==[55.53532=55.47929 + 0.05608]
Epoch 63 [55.7s]: train==[54.58751=54.53090 + 0.05658]
Epoch 64 [55.7s]: train==[53.94680=53.88979 + 0.05704]
Epoch 65 [55.6s]: train==[53.22041=53.16290 + 0.05747]
Epoch 66 [55.6s]: train==[52.28581=52.22791 + 0.05792]
Epoch 67 [55.6s]: train==[51.61816=51.55983 + 0.05836]
Epoch 68 [55.7s]: train==[50.85445=50.79557 + 0.05881]
Epoch 69 [55.7s + 194.8s]: train==[50.16910=50.10984 + 0.05924], recall=[0.13055, 0.28173], precision=[0.04058, 0.01791], hit=[0.49263, 0.72999], ndcg=[0.11380, 0.15815]
save the weights in path:  model/69.pkl
Epoch 70 [55.6s]: train==[49.77240=49.71266 + 0.05968]
Epoch 71 [55.5s]: train==[49.10279=49.04266 + 0.06012]
Epoch 72 [55.5s]: train==[48.40614=48.34565 + 0.06052]
Epoch 73 [55.6s]: train==[47.46129=47.40023 + 0.06098]
Epoch 74 [55.6s]: train==[47.02311=46.96170 + 0.06139]
Epoch 75 [55.6s]: train==[46.51810=46.45630 + 0.06181]
Epoch 76 [55.5s]: train==[45.81126=45.74894 + 0.06226]
Epoch 77 [55.5s]: train==[45.22916=45.16648 + 0.06265]
Epoch 78 [55.5s]: train==[44.85696=44.79392 + 0.06306]
Epoch 79 [55.5s + 194.1s]: train==[44.49360=44.43009 + 0.06352], recall=[0.13325, 0.28547], precision=[0.04141, 0.01817], hit=[0.49973, 0.73464], ndcg=[0.11587, 0.16045]
save the weights in path:  model/79.pkl
Epoch 80 [55.7s]: train==[43.62029=43.55635 + 0.06393]
Epoch 81 [55.7s]: train==[43.19944=43.13517 + 0.06430]
Epoch 82 [55.7s]: train==[42.97004=42.90535 + 0.06473]
Epoch 83 [55.6s]: train==[42.15358=42.08844 + 0.06514]
Epoch 84 [55.7s]: train==[41.80303=41.73752 + 0.06554]
Epoch 85 [55.7s]: train==[41.44065=41.37468 + 0.06594]
Epoch 86 [55.7s]: train==[40.94766=40.88133 + 0.06633]
Epoch 87 [55.7s]: train==[40.48489=40.41820 + 0.06672]
Epoch 88 [55.7s]: train==[39.90154=39.83447 + 0.06709]
Epoch 89 [55.6s + 194.7s]: train==[39.87759=39.81010 + 0.06750], recall=[0.13511, 0.29001], precision=[0.04192, 0.01841], hit=[0.50315, 0.74007], ndcg=[0.11750, 0.16281]
save the weights in path:  model/89.pkl
Epoch 90 [55.6s]: train==[39.04403=38.97617 + 0.06787]
Epoch 91 [55.6s]: train==[38.83527=38.76700 + 0.06829]
Epoch 92 [55.6s]: train==[38.48101=38.41237 + 0.06862]
Epoch 93 [55.5s]: train==[37.96366=37.89462 + 0.06903]
Epoch 94 [55.6s]: train==[37.91447=37.84510 + 0.06937]
Epoch 95 [55.6s]: train==[37.10428=37.03455 + 0.06974]
Epoch 96 [55.6s]: train==[36.53094=36.46080 + 0.07015]
Epoch 97 [55.5s]: train==[36.52163=36.45115 + 0.07049]
Epoch 98 [55.6s]: train==[36.14775=36.07687 + 0.07086]
Epoch 99 [55.5s + 195.8s]: train==[35.90263=35.83140 + 0.07124], recall=[0.13627, 0.29277], precision=[0.04226, 0.01861], hit=[0.50683, 0.74241], ndcg=[0.11860, 0.16445]
save the weights in path:  model/99.pkl
Epoch 100 [55.7s]: train==[35.50132=35.42978 + 0.07159]
Epoch 101 [55.6s]: train==[35.04802=34.97609 + 0.07196]
Epoch 102 [55.6s]: train==[34.93169=34.85938 + 0.07232]
Epoch 103 [55.6s]: train==[34.79682=34.72418 + 0.07267]
Epoch 104 [55.6s]: train==[33.93500=33.86196 + 0.07305]
Epoch 105 [55.6s]: train==[33.86863=33.79522 + 0.07340]
Epoch 106 [55.6s]: train==[33.73887=33.66515 + 0.07374]
Epoch 107 [55.6s]: train==[33.28534=33.21126 + 0.07407]
Epoch 108 [55.6s]: train==[32.86540=32.79097 + 0.07445]
Epoch 109 [55.6s + 196.9s]: train==[32.98528=32.91050 + 0.07478], recall=[0.13876, 0.29726], precision=[0.04287, 0.01883], hit=[0.51122, 0.74747], ndcg=[0.11998, 0.16635]
save the weights in path:  model/109.pkl
Epoch 110 [55.6s]: train==[32.29169=32.21652 + 0.07516]
Epoch 111 [55.6s]: train==[32.19059=32.11513 + 0.07548]
Epoch 112 [55.6s]: train==[32.15238=32.07656 + 0.07584]
Epoch 113 [55.6s]: train==[31.71887=31.64270 + 0.07616]
Epoch 114 [55.5s]: train==[31.56916=31.49268 + 0.07648]
Epoch 115 [55.6s]: train==[31.02398=30.94714 + 0.07683]
Epoch 116 [55.6s]: train==[30.99518=30.91807 + 0.07713]
Epoch 117 [55.6s]: train==[30.75393=30.67645 + 0.07750]
Epoch 118 [55.6s]: train==[30.55261=30.47476 + 0.07782]
Epoch 119 [55.7s + 195.7s]: train==[30.30476=30.22662 + 0.07813], recall=[0.13945, 0.30039], precision=[0.04318, 0.01902], hit=[0.51306, 0.75049], ndcg=[0.12103, 0.16812]
save the weights in path:  model/119.pkl
Epoch 120 [55.6s]: train==[29.78527=29.70682 + 0.07847]
Epoch 121 [55.5s]: train==[29.74340=29.66460 + 0.07878]
Epoch 122 [55.5s]: train==[29.59563=29.51652 + 0.07910]
Epoch 123 [55.5s]: train==[29.17373=29.09427 + 0.07947]
Epoch 124 [55.5s]: train==[29.38280=29.30305 + 0.07976]
Epoch 125 [55.6s]: train==[28.84561=28.76551 + 0.08009]
Epoch 126 [55.5s]: train==[28.85299=28.77260 + 0.08038]
Epoch 127 [55.5s]: train==[28.60495=28.52432 + 0.08066]
Epoch 128 [55.5s]: train==[28.31959=28.23857 + 0.08101]
Epoch 129 [55.5s + 192.9s]: train==[28.01187=27.93055 + 0.08134], recall=[0.14167, 0.30304], precision=[0.04365, 0.01918], hit=[0.51748, 0.75250], ndcg=[0.12242, 0.16969]
save the weights in path:  model/129.pkl
Epoch 130 [55.7s]: train==[28.01780=27.93612 + 0.08167]
Epoch 131 [55.6s]: train==[27.60601=27.52407 + 0.08193]
Epoch 132 [55.6s]: train==[27.60495=27.52266 + 0.08230]
Epoch 133 [55.6s]: train==[27.49892=27.41637 + 0.08257]
Epoch 134 [55.6s]: train==[27.22035=27.13749 + 0.08286]
Epoch 135 [55.6s]: train==[26.89074=26.80754 + 0.08319]
Epoch 136 [55.7s]: train==[26.76606=26.68261 + 0.08348]
Epoch 137 [55.7s]: train==[26.74848=26.66470 + 0.08376]
Epoch 138 [55.7s]: train==[26.33254=26.24846 + 0.08406]
Epoch 139 [55.6s + 195.8s]: train==[26.28096=26.19664 + 0.08434], recall=[0.14224, 0.30536], precision=[0.04391, 0.01930], hit=[0.51879, 0.75474], ndcg=[0.12273, 0.17053]
save the weights in path:  model/139.pkl
Epoch 140 [56.4s]: train==[26.21929=26.13462 + 0.08466]
Epoch 141 [56.1s]: train==[25.91432=25.82932 + 0.08501]
Epoch 142 [56.3s]: train==[25.83417=25.74890 + 0.08526]
Epoch 143 [55.7s]: train==[25.71249=25.62697 + 0.08554]
Epoch 144 [55.6s]: train==[25.35692=25.27110 + 0.08582]
Epoch 145 [55.6s]: train==[25.26586=25.17973 + 0.08614]
Epoch 146 [55.5s]: train==[25.00677=24.92033 + 0.08644]
Epoch 147 [55.6s]: train==[24.72652=24.63977 + 0.08674]
Epoch 148 [55.6s]: train==[24.94947=24.86244 + 0.08704]
Epoch 149 [55.6s + 196.7s]: train==[24.59413=24.50686 + 0.08728], recall=[0.14329, 0.30722], precision=[0.04412, 0.01944], hit=[0.52097, 0.75584], ndcg=[0.12372, 0.17187]
save the weights in path:  model/149.pkl
Epoch 150 [68.5s]: train==[24.46466=24.37708 + 0.08761]
Epoch 151 [69.1s]: train==[24.55402=24.46614 + 0.08786]
Epoch 152 [69.2s]: train==[24.35700=24.26884 + 0.08816]
Epoch 153 [69.4s]: train==[24.16518=24.07675 + 0.08845]
Epoch 154 [67.8s]: train==[23.98047=23.89172 + 0.08873]
Epoch 155 [58.1s]: train==[24.02263=23.93361 + 0.08902]
Epoch 156 [67.0s]: train==[23.66951=23.58021 + 0.08927]
Epoch 157 [69.3s]: train==[23.63677=23.54723 + 0.08955]
Epoch 158 [69.3s]: train==[23.39901=23.30914 + 0.08986]
Epoch 159 [69.3s + 207.6s]: train==[23.36937=23.27929 + 0.09008], recall=[0.14447, 0.30982], precision=[0.04457, 0.01961], hit=[0.52358, 0.75762], ndcg=[0.12481, 0.17336]
save the weights in path:  model/159.pkl
Epoch 160 [69.2s]: train==[23.47308=23.38271 + 0.09036]
Epoch 161 [69.3s]: train==[23.15969=23.06904 + 0.09067]
Epoch 162 [69.4s]: train==[23.26354=23.17261 + 0.09093]
Epoch 163 [65.1s]: train==[23.02622=22.93508 + 0.09116]
Epoch 164 [58.7s]: train==[22.67205=22.58057 + 0.09148]
Epoch 165 [69.4s]: train==[22.59395=22.50220 + 0.09175]
Epoch 166 [69.4s]: train==[22.49173=22.39975 + 0.09198]
Epoch 167 [69.4s]: train==[22.39703=22.30473 + 0.09228]
Epoch 168 [69.4s]: train==[22.12485=22.03235 + 0.09250]
Epoch 169 [66.8s + 205.8s]: train==[22.31854=22.22575 + 0.09277], recall=[0.14506, 0.31167], precision=[0.04466, 0.01973], hit=[0.52321, 0.75745], ndcg=[0.12513, 0.17411]
save the weights in path:  model/169.pkl
Epoch 170 [69.2s]: train==[21.79519=21.70212 + 0.09307]
Epoch 171 [69.3s]: train==[21.92234=21.82901 + 0.09333]
Epoch 172 [62.6s]: train==[21.84521=21.75156 + 0.09365]
Epoch 173 [61.1s]: train==[21.75822=21.66438 + 0.09384]
Epoch 174 [69.4s]: train==[21.64133=21.54722 + 0.09411]
Epoch 175 [69.4s]: train==[21.23021=21.13583 + 0.09438]
Epoch 176 [69.3s]: train==[21.31801=21.22336 + 0.09464]
Epoch 177 [69.4s]: train==[21.33176=21.23684 + 0.09490]
Epoch 178 [64.2s]: train==[21.27108=21.17591 + 0.09517]
Epoch 179 [59.5s + 199.1s]: train==[21.08430=20.98891 + 0.09541], recall=[0.14577, 0.31272], precision=[0.04497, 0.01978], hit=[0.52549, 0.76033], ndcg=[0.12585, 0.17488]
save the weights in path:  model/179.pkl
Epoch 180 [69.2s]: train==[20.80277=20.70714 + 0.09563]
Epoch 181 [59.9s]: train==[20.82540=20.72952 + 0.09588]
Epoch 182 [64.0s]: train==[20.74550=20.64932 + 0.09615]
Epoch 183 [69.4s]: train==[20.54355=20.44716 + 0.09643]
Epoch 184 [69.4s]: train==[20.53159=20.43491 + 0.09668]
Epoch 185 [69.4s]: train==[20.71714=20.62019 + 0.09695]
Epoch 186 [69.4s]: train==[20.39958=20.30243 + 0.09716]
Epoch 187 [61.4s]: train==[20.24019=20.14275 + 0.09745]
Epoch 188 [62.2s]: train==[20.30680=20.20915 + 0.09767]
Epoch 189 [69.4s + 199.5s]: train==[20.02591=19.92796 + 0.09795], recall=[0.14650, 0.31428], precision=[0.04507, 0.01988], hit=[0.52509, 0.76100], ndcg=[0.12611, 0.17545]
save the weights in path:  model/189.pkl
Epoch 190 [58.4s]: train==[19.97899=19.88078 + 0.09821]
Epoch 191 [66.2s]: train==[19.89351=19.79508 + 0.09843]
Epoch 192 [69.4s]: train==[19.90920=19.81055 + 0.09866]
Epoch 193 [69.3s]: train==[19.62221=19.52326 + 0.09895]
Epoch 194 [69.4s]: train==[19.60901=19.50984 + 0.09917]
Epoch 195 [69.3s]: train==[19.40484=19.30550 + 0.09936]
Epoch 196 [59.0s]: train==[19.51504=19.41538 + 0.09965]
Epoch 197 [64.8s]: train==[19.26628=19.16644 + 0.09985]
Epoch 198 [69.4s]: train==[19.27599=19.17589 + 0.10011]
Epoch 199 [69.4s + 206.2s]: train==[19.37243=19.27208 + 0.10036], recall=[0.14693, 0.31655], precision=[0.04533, 0.02000], hit=[0.52679, 0.76234], ndcg=[0.12660, 0.17646]
save the weights in path:  model/199.pkl
Epoch 200 [68.5s]: train==[19.00498=18.90435 + 0.10063]
Epoch 201 [69.2s]: train==[19.00021=18.89937 + 0.10083]
Epoch 202 [69.3s]: train==[18.86273=18.76161 + 0.10111]
Epoch 203 [69.4s]: train==[18.63849=18.53715 + 0.10134]
Epoch 204 [67.7s]: train==[18.82091=18.71934 + 0.10157]
Epoch 205 [58.2s]: train==[18.76085=18.65909 + 0.10176]
Epoch 206 [67.1s]: train==[18.47148=18.36941 + 0.10205]
Epoch 207 [69.4s]: train==[18.60222=18.49992 + 0.10228]
Epoch 208 [69.4s]: train==[18.50873=18.40622 + 0.10252]
Epoch 209 [69.4s + 207.2s]: train==[18.49551=18.39278 + 0.10274], recall=[0.14807, 0.31833], precision=[0.04558, 0.02008], hit=[0.52907, 0.76465], ndcg=[0.12738, 0.17740]
save the weights in path:  model/209.pkl
Epoch 210 [69.1s]: train==[18.38564=18.28265 + 0.10300]
Epoch 211 [69.4s]: train==[18.22724=18.12404 + 0.10320]
Epoch 212 [69.4s]: train==[18.08409=17.98067 + 0.10341]
Epoch 213 [65.5s]: train==[17.93998=17.83635 + 0.10365]
Epoch 214 [58.3s]: train==[18.05466=17.95082 + 0.10383]
Epoch 215 [69.3s]: train==[18.01194=17.90786 + 0.10409]
Epoch 216 [69.4s]: train==[17.98426=17.87996 + 0.10430]
Epoch 217 [69.4s]: train==[17.63272=17.52816 + 0.10455]
Epoch 218 [69.4s]: train==[17.83654=17.73172 + 0.10481]
Epoch 219 [67.3s + 203.9s]: train==[17.70797=17.60294 + 0.10502], recall=[0.14910, 0.31935], precision=[0.04589, 0.02014], hit=[0.53034, 0.76593], ndcg=[0.12805, 0.17806]
save the weights in path:  model/219.pkl
Epoch 220 [69.2s]: train==[17.55494=17.44970 + 0.10526]
Epoch 221 [69.4s]: train==[17.52306=17.41759 + 0.10547]
Epoch 222 [63.7s]: train==[17.48969=17.38398 + 0.10571]
Epoch 223 [59.9s]: train==[17.39531=17.28941 + 0.10591]
Epoch 224 [69.4s]: train==[17.21877=17.11263 + 0.10614]
Epoch 225 [69.4s]: train==[17.26140=17.15504 + 0.10636]
Epoch 226 [69.4s]: train==[17.28984=17.18323 + 0.10660]
Epoch 227 [69.4s]: train==[17.26980=17.16297 + 0.10682]
Epoch 228 [65.5s]: train==[17.20937=17.10234 + 0.10703]
Epoch 229 [58.2s + 196.5s]: train==[16.91563=16.80840 + 0.10724], recall=[0.14939, 0.32097], precision=[0.04596, 0.02024], hit=[0.53041, 0.76599], ndcg=[0.12832, 0.17875]
save the weights in path:  model/229.pkl
Epoch 230 [69.3s]: train==[16.97347=16.86601 + 0.10746]
Epoch 231 [61.9s]: train==[16.68708=16.57937 + 0.10770]
Epoch 232 [62.0s]: train==[16.70382=16.59595 + 0.10788]
Epoch 233 [69.4s]: train==[16.64958=16.54146 + 0.10812]
Epoch 234 [69.4s]: train==[16.52839=16.42004 + 0.10836]
Epoch 235 [69.4s]: train==[16.67065=16.56211 + 0.10855]
Epoch 236 [69.4s]: train==[16.60280=16.49402 + 0.10878]
Epoch 237 [63.4s]: train==[16.60825=16.49925 + 0.10900]
Epoch 238 [60.3s]: train==[16.40417=16.29498 + 0.10920]
Epoch 239 [69.4s + 196.5s]: train==[16.13950=16.03004 + 0.10946], recall=[0.15060, 0.32228], precision=[0.04636, 0.02032], hit=[0.53456, 0.76760], ndcg=[0.12905, 0.17947]
save the weights in path:  model/239.pkl
Epoch 240 [59.6s]: train==[16.30202=16.19240 + 0.10961]
Epoch 241 [63.9s]: train==[16.16611=16.05628 + 0.10984]
Epoch 242 [69.4s]: train==[16.19358=16.08354 + 0.11003]
Epoch 243 [69.4s]: train==[16.14655=16.03627 + 0.11028]
Epoch 244 [69.4s]: train==[16.09139=15.98091 + 0.11048]
Epoch 245 [69.4s]: train==[16.01182=15.90119 + 0.11064]
Epoch 246 [61.4s]: train==[16.09975=15.98884 + 0.11091]
Epoch 247 [62.1s]: train==[15.94434=15.83323 + 0.11111]
Epoch 248 [69.4s]: train==[15.78388=15.67253 + 0.11136]
Epoch 249 [69.4s + 206.0s]: train==[15.75310=15.64159 + 0.11151], recall=[0.15161, 0.32261], precision=[0.04664, 0.02037], hit=[0.53470, 0.76837], ndcg=[0.12976, 0.18000]
save the weights in path:  model/249.pkl
Epoch 250 [66.3s]: train==[15.58059=15.46886 + 0.11173]
Epoch 251 [69.3s]: train==[15.63062=15.51870 + 0.11192]
Epoch 252 [69.4s]: train==[15.66632=15.55419 + 0.11214]
Epoch 253 [69.4s]: train==[15.50759=15.39524 + 0.11234]
Epoch 254 [69.4s]: train==[15.47596=15.36339 + 0.11257]
Epoch 255 [58.9s]: train==[15.32440=15.21167 + 0.11272]
Epoch 256 [64.7s]: train==[15.31739=15.20441 + 0.11298]
Epoch 257 [69.4s]: train==[15.20525=15.09209 + 0.11316]
Epoch 258 [69.4s]: train==[15.07312=14.95978 + 0.11335]
Epoch 259 [69.4s + 208.2s]: train==[15.12629=15.01270 + 0.11357], recall=[0.15138, 0.32422], precision=[0.04657, 0.02046], hit=[0.53386, 0.76790], ndcg=[0.12957, 0.18038]
Epoch 260 [69.3s]: train==[15.31727=15.20349 + 0.11378]
Epoch 261 [69.4s]: train==[15.24764=15.13364 + 0.11400]
Epoch 262 [69.4s]: train==[15.24096=15.12678 + 0.11419]
Epoch 263 [67.4s]: train==[15.02009=14.90571 + 0.11437]
Epoch 264 [58.2s]: train==[14.94320=14.82863 + 0.11459]
Epoch 265 [67.6s]: train==[15.03446=14.91969 + 0.11477]
Epoch 266 [69.4s]: train==[14.82776=14.71280 + 0.11496]
Epoch 267 [69.4s]: train==[14.99813=14.88296 + 0.11517]
Epoch 268 [69.4s]: train==[14.72450=14.60916 + 0.11535]
Epoch 269 [68.9s + 206.1s]: train==[14.89456=14.77896 + 0.11559], recall=[0.15254, 0.32529], precision=[0.04695, 0.02049], hit=[0.53651, 0.76968], ndcg=[0.13026, 0.18097]
save the weights in path:  model/269.pkl
Epoch 270 [69.3s]: train==[14.73163=14.61588 + 0.11575]
Epoch 271 [69.4s]: train==[14.45822=14.34224 + 0.11598]
Epoch 272 [65.1s]: train==[14.64558=14.52943 + 0.11616]
Epoch 273 [58.6s]: train==[14.50271=14.38634 + 0.11636]
Epoch 274 [69.4s]: train==[14.48598=14.36942 + 0.11657]
Epoch 275 [69.4s]: train==[14.50778=14.39107 + 0.11672]
Epoch 276 [69.4s]: train==[14.43592=14.31898 + 0.11694]
Epoch 277 [69.4s]: train==[14.39425=14.27709 + 0.11715]
Epoch 278 [66.9s]: train==[14.40241=14.28507 + 0.11734]
Epoch 279 [58.3s + 198.0s]: train==[14.32268=14.20515 + 0.11753], recall=[0.15303, 0.32615], precision=[0.04701, 0.02055], hit=[0.53724, 0.77031], ndcg=[0.13038, 0.18125]
save the weights in path:  model/279.pkl
Epoch 280 [69.4s]: train==[14.33080=14.21311 + 0.11771]
Epoch 281 [63.0s]: train==[14.26031=14.14245 + 0.11788]
Epoch 282 [60.6s]: train==[14.10835=13.99022 + 0.11813]
Epoch 283 [69.4s]: train==[13.91976=13.80147 + 0.11829]
Epoch 284 [69.4s]: train==[14.02559=13.90710 + 0.11849]
Epoch 285 [69.4s]: train==[14.12153=14.00285 + 0.11867]
Epoch 286 [69.4s]: train==[13.93093=13.81208 + 0.11885]
Epoch 287 [64.7s]: train==[14.02750=13.90848 + 0.11902]
Epoch 288 [58.8s]: train==[14.03139=13.91211 + 0.11927]
Epoch 289 [69.4s + 198.3s]: train==[13.86251=13.74307 + 0.11944], recall=[0.15275, 0.32616], precision=[0.04689, 0.02057], hit=[0.53557, 0.77095], ndcg=[0.13030, 0.18137]
Epoch 290 [60.7s]: train==[13.59236=13.47271 + 0.11965]
Epoch 291 [62.9s]: train==[13.90866=13.78885 + 0.11981]
Epoch 292 [69.3s]: train==[13.57098=13.45101 + 0.11998]
Epoch 293 [69.3s]: train==[13.57014=13.44993 + 0.12020]
Epoch 294 [69.4s]: train==[13.48047=13.36009 + 0.12039]
Epoch 295 [69.5s]: train==[13.52223=13.40161 + 0.12061]
Epoch 296 [62.5s]: train==[13.38261=13.26187 + 0.12075]
Epoch 297 [61.1s]: train==[13.52341=13.40245 + 0.12094]
Epoch 298 [69.4s]: train==[13.27989=13.15877 + 0.12113]
Epoch 299 [69.4s + 204.4s]: train==[13.34107=13.21978 + 0.12129], recall=[0.15254, 0.32700], precision=[0.04686, 0.02061], hit=[0.53487, 0.77165], ndcg=[0.13023, 0.18155]
Epoch 300 [56.6s]: train==[13.26524=13.14375 + 0.12149]
Epoch 301 [55.7s]: train==[13.34701=13.22533 + 0.12166]
Epoch 302 [55.6s]: train==[13.32464=13.20278 + 0.12186]
Epoch 303 [55.6s]: train==[13.36789=13.24585 + 0.12204]
Epoch 304 [55.6s]: train==[13.44998=13.32773 + 0.12224]
Epoch 305 [55.6s]: train==[13.10960=12.98717 + 0.12242]
Epoch 306 [55.6s]: train==[13.26233=13.13976 + 0.12258]
Epoch 307 [55.6s]: train==[13.18512=13.06237 + 0.12276]
Epoch 308 [55.7s]: train==[13.23300=13.11006 + 0.12294]
Epoch 309 [55.7s + 194.6s]: train==[13.02210=12.89899 + 0.12312], recall=[0.15354, 0.32760], precision=[0.04716, 0.02070], hit=[0.53798, 0.77031], ndcg=[0.13088, 0.18213]
save the weights in path:  model/309.pkl
Epoch 310 [55.6s]: train==[13.30202=13.17877 + 0.12327]
Epoch 311 [55.6s]: train==[13.14912=13.02567 + 0.12347]
Epoch 312 [55.6s]: train==[12.88727=12.76356 + 0.12372]
Epoch 313 [55.6s]: train==[13.22253=13.09872 + 0.12381]
Epoch 314 [55.6s]: train==[12.79683=12.67285 + 0.12399]
Epoch 315 [55.6s]: train==[12.87777=12.75356 + 0.12419]
Epoch 316 [55.6s]: train==[12.76795=12.64359 + 0.12436]
Epoch 317 [55.6s]: train==[12.74186=12.61736 + 0.12450]
Epoch 318 [55.7s]: train==[12.85926=12.73455 + 0.12472]
Epoch 319 [55.6s + 193.7s]: train==[12.69827=12.57341 + 0.12487], recall=[0.15399, 0.32849], precision=[0.04723, 0.02070], hit=[0.53919, 0.77159], ndcg=[0.13139, 0.18272]
save the weights in path:  model/319.pkl
Epoch 320 [55.6s]: train==[12.72991=12.60485 + 0.12508]
Epoch 321 [55.6s]: train==[12.46464=12.33939 + 0.12527]
Epoch 322 [55.6s]: train==[12.51435=12.38891 + 0.12544]
Epoch 323 [55.6s]: train==[12.54478=12.41918 + 0.12561]
Epoch 324 [55.6s]: train==[12.44448=12.31870 + 0.12579]
Epoch 325 [55.6s]: train==[12.57135=12.44538 + 0.12598]
Epoch 326 [55.6s]: train==[12.35412=12.22796 + 0.12615]
Epoch 327 [55.6s]: train==[12.51190=12.38555 + 0.12635]
Epoch 328 [55.6s]: train==[12.33640=12.20989 + 0.12650]
Epoch 329 [55.6s + 196.0s]: train==[12.17656=12.04987 + 0.12669], recall=[0.15378, 0.32908], precision=[0.04722, 0.02078], hit=[0.53838, 0.77209], ndcg=[0.13114, 0.18282]
Epoch 330 [55.6s]: train==[12.34177=12.21490 + 0.12687]
Epoch 331 [55.6s]: train==[12.11922=11.99222 + 0.12700]
Epoch 332 [55.5s]: train==[12.24042=12.11326 + 0.12715]
Epoch 333 [55.6s]: train==[12.30303=12.17567 + 0.12735]
Epoch 334 [55.5s]: train==[12.17422=12.04670 + 0.12751]
Epoch 335 [55.5s]: train==[12.07101=11.94331 + 0.12770]
Epoch 336 [55.6s]: train==[12.27273=12.14485 + 0.12788]
Epoch 337 [55.5s]: train==[11.99014=11.86210 + 0.12804]
Epoch 338 [55.5s]: train==[12.04310=11.91486 + 0.12824]
Epoch 339 [55.6s + 193.9s]: train==[11.92394=11.79557 + 0.12839], recall=[0.15440, 0.32976], precision=[0.04742, 0.02082], hit=[0.53825, 0.77319], ndcg=[0.13154, 0.18320]
save the weights in path:  model/339.pkl
Epoch 340 [55.7s]: train==[11.87195=11.74340 + 0.12856]
Epoch 341 [55.7s]: train==[11.99539=11.86665 + 0.12874]
Epoch 342 [55.8s]: train==[11.97576=11.84687 + 0.12890]
Epoch 343 [55.7s]: train==[11.95100=11.82195 + 0.12905]
Epoch 344 [55.7s]: train==[11.97022=11.84098 + 0.12923]
Epoch 345 [55.7s]: train==[11.61318=11.48377 + 0.12941]
Epoch 346 [55.7s]: train==[11.85413=11.72457 + 0.12957]
Epoch 347 [55.7s]: train==[11.77318=11.64346 + 0.12972]
Epoch 348 [55.7s]: train==[11.71866=11.58873 + 0.12993]
Epoch 349 [55.7s + 194.0s]: train==[11.81401=11.68388 + 0.13012], recall=[0.15454, 0.32998], precision=[0.04748, 0.02084], hit=[0.53805, 0.77333], ndcg=[0.13176, 0.18351]
save the weights in path:  model/349.pkl
Epoch 350 [55.5s]: train==[11.67969=11.54943 + 0.13026]
Epoch 351 [55.4s]: train==[11.30364=11.17320 + 0.13044]
Epoch 352 [55.5s]: train==[11.50849=11.37792 + 0.13057]
Epoch 353 [55.5s]: train==[11.52253=11.39177 + 0.13075]
Epoch 354 [55.5s]: train==[11.44260=11.31169 + 0.13092]
Epoch 355 [55.5s]: train==[11.51324=11.38217 + 0.13107]
Epoch 356 [55.5s]: train==[11.38568=11.25447 + 0.13122]
Epoch 357 [55.5s]: train==[11.28273=11.15131 + 0.13141]
Epoch 358 [55.5s]: train==[11.31673=11.18510 + 0.13162]
Epoch 359 [55.5s + 194.9s]: train==[11.44204=11.31028 + 0.13176], recall=[0.15496, 0.33066], precision=[0.04762, 0.02088], hit=[0.53892, 0.77226], ndcg=[0.13212, 0.18391]
save the weights in path:  model/359.pkl
Epoch 360 [55.6s]: train==[11.35021=11.21828 + 0.13192]
Epoch 361 [55.5s]: train==[11.38721=11.25515 + 0.13206]
Epoch 362 [55.5s]: train==[11.28724=11.15504 + 0.13221]
Epoch 363 [55.5s]: train==[11.27499=11.14264 + 0.13235]
Epoch 364 [55.5s]: train==[11.21212=11.07954 + 0.13258]
Epoch 365 [55.5s]: train==[11.21292=11.08019 + 0.13273]
Epoch 366 [55.6s]: train==[11.03282=10.89992 + 0.13290]
Epoch 367 [55.6s]: train==[11.04048=10.90744 + 0.13304]
Epoch 368 [55.5s]: train==[11.00437=10.87118 + 0.13319]
Epoch 369 [55.5s + 194.0s]: train==[11.11107=10.97768 + 0.13339], recall=[0.15502, 0.33070], precision=[0.04762, 0.02089], hit=[0.53862, 0.77252], ndcg=[0.13241, 0.18424]
save the weights in path:  model/369.pkl
Epoch 370 [55.6s]: train==[11.04667=10.91310 + 0.13358]
Epoch 371 [55.5s]: train==[11.00423=10.87052 + 0.13370]
Epoch 372 [55.6s]: train==[11.05905=10.92520 + 0.13385]
Epoch 373 [55.6s]: train==[10.96999=10.83599 + 0.13399]
Epoch 374 [55.6s]: train==[11.00842=10.87426 + 0.13416]
Epoch 375 [55.6s]: train==[11.05831=10.92395 + 0.13436]
Epoch 376 [55.6s]: train==[10.97698=10.84245 + 0.13454]
Epoch 377 [55.6s]: train==[10.85985=10.72513 + 0.13471]
Epoch 378 [55.5s]: train==[10.76060=10.62579 + 0.13480]
Epoch 379 [55.6s + 193.8s]: train==[10.79205=10.65707 + 0.13498], recall=[0.15577, 0.33128], precision=[0.04766, 0.02089], hit=[0.53888, 0.77242], ndcg=[0.13215, 0.18397]
save the weights in path:  model/379.pkl
Epoch 380 [55.6s]: train==[10.79622=10.66109 + 0.13513]
Epoch 381 [55.6s]: train==[10.74607=10.61078 + 0.13529]
Epoch 382 [55.7s]: train==[10.81035=10.67488 + 0.13546]
Epoch 383 [55.6s]: train==[10.64527=10.50967 + 0.13560]
Epoch 384 [55.7s]: train==[10.67533=10.53954 + 0.13579]
Epoch 385 [55.6s]: train==[10.57267=10.43676 + 0.13591]
Epoch 386 [55.6s]: train==[10.50101=10.36497 + 0.13606]
Epoch 387 [55.6s]: train==[10.64676=10.51048 + 0.13626]
Epoch 388 [55.6s]: train==[10.62928=10.49288 + 0.13640]
Epoch 389 [55.6s + 194.2s]: train==[10.65930=10.52278 + 0.13652], recall=[0.15506, 0.33219], precision=[0.04766, 0.02097], hit=[0.53764, 0.77376], ndcg=[0.13186, 0.18407]
Epoch 390 [55.6s]: train==[10.37874=10.24203 + 0.13673]
Epoch 391 [55.5s]: train==[10.45021=10.31336 + 0.13686]
Epoch 392 [55.5s]: train==[10.41026=10.27328 + 0.13697]
Epoch 393 [55.6s]: train==[10.37612=10.23896 + 0.13717]
Epoch 394 [55.5s]: train==[10.24539=10.10805 + 0.13734]
Epoch 395 [55.5s]: train==[10.71768=10.58018 + 0.13749]
Epoch 396 [55.5s]: train==[10.38093=10.24329 + 0.13764]
Epoch 397 [55.5s]: train==[10.13543=9.99763 + 0.13780]
Epoch 398 [55.5s]: train==[10.32447=10.18652 + 0.13795]
Epoch 399 [55.5s + 195.2s]: train==[10.20420=10.06614 + 0.13806], recall=[0.15539, 0.33242], precision=[0.04777, 0.02096], hit=[0.53909, 0.77333], ndcg=[0.13252, 0.18466]
Best Iter=[37]@[31850.1]	recall=[0.15577	0.21899	0.26507	0.30061	0.33128], precision=[0.04766	0.03389	0.02751	0.02356	0.02089], hit=[0.53888	0.64412	0.70350	0.74238	0.77242], ndcg=[0.13215	0.15204	0.16572	0.17574	0.18397]

```

## 空数据处理方式

## 优化

- NGCf

#
