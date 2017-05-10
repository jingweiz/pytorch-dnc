import numpy as np

# custom modules
from utils.options import Options
from utils.factory import EnvDict, CircuitDict, AgentDict

# 0. setting up
opt = Options()
np.random.seed(opt.seed)

# 1. env     (prototype)
env_prototype     = EnvDict[opt.env_type]
# 2. circuit (prototype)
circuit_prototype = CircuitDict[opt.circuit_type]
# 3. agent
agent = AgentDict[opt.agent_type](opt.agent_params,
                                  env_prototype     = env_prototype,
                                  circuit_prototype = circuit_prototype)
# 5. fit model
if opt.mode == 1:   # train
    agent.fit_model()
elif opt.mode == 2: # test opt.model_file
    agent.test_model()
