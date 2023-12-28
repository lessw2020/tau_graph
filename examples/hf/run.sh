#PIPPY_VERBOSITY=INFO 
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_id 101 --rdzv_backend c10d  ./convbert.py