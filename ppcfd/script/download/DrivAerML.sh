# DrivAerML
# License: CC BY-SA 4.0
# Source:  DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics
# Modifed: Yes
for i in {1..500}
do
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer_Surface_Commercial/boundary_${i}_centroid.npy"
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer_Surface_Commercial/boundary_${i}_wss.npy"
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer_Surface_Commercial/boundary_${i}_p.npy"
done
