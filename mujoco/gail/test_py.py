import pickle
import numpy as np
expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb"))
demonstrations = np.array(expert_demo)
print("demonstrations.shape", demonstrations.shape)
print(expert_demo)