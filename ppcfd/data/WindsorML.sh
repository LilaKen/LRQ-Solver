# WindsorML
# License: CC BY-SA 4.0
# Source:  WindsorML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics
# Modifed: Yes
for i in {0..354}
do
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/Windsor_Surface_Commercial/boundary_${i}_centroid.npy"
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/Windsor_Surface_Commercial/boundary_${i}_wss.npy"
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/Windsor_Surface_Commercial/boundary_${i}_p.npy"
done
