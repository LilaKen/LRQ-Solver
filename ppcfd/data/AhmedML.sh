# AhmedML
# License: CC BY-SA 4.0
# Source:  AhmedML: High-Fidelity Computational Fluid Dynamics Dataset for Incompressible, Low-Speed Bluff Body Aerodynamics
# Modifed: Yes
for i in {1..500}
do
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/Ahmed_Surface_Commercial/boundary_${i}_centroid.npy"
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/Ahmed_Surface_Commercial/boundary_${i}_wss.npy"
    wget "https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/Ahmed_Surface_Commercial/boundary_${i}_p.npy"
done
