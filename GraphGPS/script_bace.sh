CUDA_VISIBLE_DEVICES=4 python main_ft.py --cfg configs/GPS/bace_ft_full.yaml --repeat 5 scaffold True 50 0.001
# CUDA_VISIBLE_DEVICES=4 python main_ft.py --cfg configs/GPS/bace_ft_full.yaml --repeat 5 scaffold True 100 0.001
CUDA_VISIBLE_DEVICES=4 python main_ft.py --cfg configs/GPS/bace_ft_full.yaml --repeat 5 scaffold True 500 0.001
CUDA_VISIBLE_DEVICES=4 python main_ft.py --cfg configs/GPS/bace_ft_full.yaml --repeat 5 scaffold False 50 0.001

# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_fm.yaml --repeat 5 scaffold False 50 1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_fm.yaml --repeat 5 scaffold False 50 0.1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_fm.yaml --repeat 5 scaffold False 50 0.01
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_fm.yaml --repeat 5 scaffold False 50 0.001
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_fm.yaml --repeat 5 scaffold False 50 0.0001

# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_full.yaml --repeat 5 scaffold False 50 0.001
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft_lp.yaml --repeat 5 scaffold False 50 0.001

# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft.yaml --repeat 5 scaffold False 50 1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft.yaml --repeat 5 scaffold False 50 0.1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft.yaml --repeat 5 scaffold False 50 0.01
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft.yaml --repeat 5 scaffold False 50 0.001
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/bace_ft.yaml --repeat 5 scaffold False 50 0.0001