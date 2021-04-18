run_square:
	nohup make square > square.log 2>&1 & 
run_gowalla:
	nohup make gowalla > gowalla.log 2>&1 & 
square:
	python3 main.py --data_path ./data/ --dataset square --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
gowalla:
	python3 main.py --data_path ./data/ --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
