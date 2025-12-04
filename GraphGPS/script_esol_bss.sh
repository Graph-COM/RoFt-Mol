CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 scaffold False 50 0.001 front
CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 scaffold False 50 0.001 mid
CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 scaffold False 50 0.001 back

CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 random False 50 0.001 front
CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 random False 50 0.001 mid
CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 random False 50 0.001 back

CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 size False 50 0.001 front
CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 size False 50 0.001 mid
CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_surg.yaml --repeat 2 size False 50 0.001 back
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_bss.yaml --repeat 2 size False 50 1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_bss.yaml --repeat 2 size False 50 0.1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_bss.yaml --repeat 2 size False 50 0.01
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_bss.yaml --repeat 2 size False 50 0.001
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_bss.yaml --repeat 2 size False 50 0.0001

# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_full.yaml --repeat 5 size False 50 0.001
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft_lp.yaml --repeat 5 size False 50 0.001

# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft.yaml --repeat 5 size False 50 1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft.yaml --repeat 5 size False 50 0.1
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft.yaml --repeat 5 size False 50 0.01
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft.yaml --repeat 5 size False 50 0.001
# CUDA_VISIBLE_DEVICES=1 python main_ft.py --cfg configs/GPS/esol_ft.yaml --repeat 5 size False 50 0.0001