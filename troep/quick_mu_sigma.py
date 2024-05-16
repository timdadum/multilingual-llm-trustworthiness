import numpy as np
"""
proprietary = np.array([0.34, 0.08, 0.06, 0.06,
                        0.12, 0.06, 0.08, 0.06,
                        0.62, 0.76, 0.94, 1,
                        0.06, 0.06, 0.04, 0.06]) # 
open_source = np.array([0.98, 0.80, 0.72, 0.46,
                        0.34, 0.32, 0.52, 0.42,
                        0.40, 0.54, 0.38, 0.06,
                        0.70, 0.82, 0.88, 0.60,
                        0.14, 0.40, 0.46, 0.18,
                        0.68, 0.20, 0.52, 0.30,
                        0.747, 0.772, 0.96, 0.98,
                        0.597, 0.709, 0.370, 0.621,
                        0.576, 0.648, 0.354, 0.589,
                        0.576, 0.452, 0.455, 0.632,
                        0.708, 0.691, 0.602, 0.690,
                        0.704, 0.751, 0.325, 0.700]) # should contain twelve
"""

proprietary = np.array([0.352, 0.386, 0.072, 0.317])
open_source = np.array([0.262, 0.112, 0.141, 0.191, 0.205, 0.248, 0.213, 0.374, 0.294, 0.183, 0.237, 0.154])

if len(proprietary) + len(open_source) != 1635:
    print(len(proprietary) + len(open_source))
    raise ValueError("Have you filled in all LLMs?")

proprietary_std = np.sqrt(proprietary.std())
proprietary_mu = proprietary.mean()
open_source_std = np.sqrt(open_source.std())
open_source_mu = open_source.mean()
print(f'Proprietary: std: {proprietary_std}, mu: {proprietary_mu}')
print(f'Open source: std: {open_source_std}, mu: {open_source_mu}')