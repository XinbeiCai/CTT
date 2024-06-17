python run_model.py --gpu_id=3 --model=MSSR --dataset="CDs_and_Vinyl" --pred='dot' \
           --ssl=1 --cl='idropwc' --tau=1 --cllmd=0.12 --aaplmd=9 --sim='dot' \
		       --n_layers=4 --clayer_num=1 --n_heads=8  \
           --attribute_predictor='linear' --aap='wi_wc_bce' --aap_gate=1 \
           --logit_num=2 --item_predictor=2 --ip_gate_mode='moe' --gate_drop=1 \
           --sc=0 --seqmc='all' --cdmc='all' \
           --train_batch_size=128 --pooling_mode='mean' --config_files="configs/CDs_and_Vinyl.yaml" \
           --attribute_hidden_size=[256] --ada_fuse=1 --fusion_type=gate --result_file='result.txt';