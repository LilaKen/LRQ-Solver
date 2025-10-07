import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

# Define the LHS parameters number
dim = 5

# Create a Latin Hypercube Sample
lhs = qmc.LatinHypercube(dim)

# Generate sample
sample = lhs.random(n=500)  # 'n' is the number of points


sample[:, 0] = (
    sample[:, 0]
) * 0.5 - 0.1  #   trunklid_angle          -8  ~ 21.8   dgree
sample[:, 1] = (
    sample[:, 1]
) * 1.8 - 0.9  #   ramp_angle              -8  ~ 16     dgree
sample[:, 2] = (
    sample[:, 2]
) * 0.8 - 0.2  #   diffusor_angle          -8  ~ 16     dgree
sample[:, 3] = (sample[:, 3]) * 0.4 - 0.1  #   front_bumper_length     -25 ~ 60     mm
sample[:, 4] = (sample[:, 4]) * 0.4 - 0.2  #   trunklid_length         -40 ~ 40     mm

print(sample[:10])
# Plot the points
plt.scatter(sample[:, 0], sample[:, 1])
plt.title("Latin Hypercube Sampling")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig("lhs.png")
np.savetxt("lhs_parameters.csv", sample, delimiter=",")
