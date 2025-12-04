
CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 scaffold False 50 0.001
CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 random False 50 0.001
CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 size False 50 0.001

# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 scaffold True 50 0.001
# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 scaffold True 100 0.001
# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 scaffold True 500 0.001

# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 size True 50 0.001
# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 size True 100 0.001
# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 size True 500 0.001

# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 random True 50 0.001
# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 random True 100 0.001
# CUDA_VISIBLE_DEVICES=2 python main_ft.py --cfg configs/GPS/bace_ft_lpft.yaml --repeat 5 random True 500 0.001
