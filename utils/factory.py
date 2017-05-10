from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.envs.copy_env import CopyEnv
from core.envs.repeat_copy_env import RepeatCopyEnv
EnvDict = {"copy":        CopyEnv,
           "repeat-copy": RepeatCopyEnv}

from core.circuits.ntm_circuit import NTMCircuit
from core.circuits.dnc_circuit import DNCCircuit
CircuitDict = {"none": None,
               "ntm":  NTMCircuit,  # lstm_controller + static_accessor
               "dnc":  DNCCircuit}  # lstm_controller + dynamic_accessor

from core.agents.empty_agent import EmptyAgent
from core.agents.sl_agent    import SLAgent
AgentDict = {"empty": EmptyAgent,   # to test integration of new envs, contains only the most basic control loop
             "sl":    SLAgent}      # for supervised learning tasks
