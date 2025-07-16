# Action Learning Engine

`ActionLearningEngine` learns optimal action sequences from `PathwayDB` histories. When `stable_baselines3` is installed you can choose different reinforcement learning algorithms.

```python
from menace.neuroplasticity import PathwayDB
from menace.menace_memory_manager import MenaceMemoryManager
from menace.code_database import CodeDB
from menace.unified_learning_engine import UnifiedLearningEngine
from menace.action_learning_engine import ActionLearningEngine

pdb = PathwayDB("p.db")
roi = DummyROIDB()
code_db = CodeDB("c.db")
mm = MenaceMemoryManager("m.db")
ule = UnifiedLearningEngine(pdb, mm, code_db, roi)

# Train using SAC with a custom learning rate
engine = ActionLearningEngine(
    pdb,
    roi,
    code_db,
    ule,
    algo="sac",
    algo_kwargs={"learning_rate": 5e-4},
)
engine.train()
```

Supported algorithms include `DQN`, `PPO`, `A2C`, `SAC` and `TD3`. Pass `train_steps` to control the number of timesteps for `.learn()`.
