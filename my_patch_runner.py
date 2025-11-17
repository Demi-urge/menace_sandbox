from pathlib import Path
import sys
import __main__

from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.code_database import CodeDB
from menace_sandbox.coding_bot_interface import fallback_helper_manager, prepare_pipeline_for_bootstrap
from menace_sandbox.data_bot import DataBot
from menace_sandbox.menace_memory_manager import MenaceMemoryManager
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
from menace_sandbox.self_coding_engine import SelfCodingEngine
from menace_sandbox.self_coding_manager import SelfCodingManager, internalize_coding_bot
from menace_sandbox.self_coding_thresholds import get_thresholds
from menace_sandbox.threshold_service import ThresholdService
import menace_sandbox.patch_application as patch_application

# Inputs you control
module_path = "path/to/file.py"
description = "Describe the change"

# Build engine + context builder
registry = BotRegistry()
data_bot = DataBot(start_server=False)
engine = SelfCodingEngine(CodeDB(), MenaceMemoryManager())
builder = engine.context_builder

# Install a bootstrap sentinel manager while the pipeline is constructed
with fallback_helper_manager(bot_registry=registry, data_bot=data_bot) as bootstrap_manager:
    pipeline, promote_pipeline = prepare_pipeline_for_bootstrap(
        pipeline_cls=ModelAutomationPipeline,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
        bootstrap_runtime_manager=bootstrap_manager,
        manager=bootstrap_manager,
    )

# Internalize a real SelfCodingManager and promote helpers
bot_name = Path(module_path).stem
thresholds = get_thresholds(bot_name)
manager: SelfCodingManager = internalize_coding_bot(
    bot_name,
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    roi_threshold=thresholds.roi_drop,
    error_threshold=thresholds.error_increase,
    test_failure_threshold=thresholds.test_failure_increase,
    threshold_service=ThresholdService(),
)
promote_pipeline(manager)

# Expose the active manager to patch_application
__main__.manager = manager
sys.argv = [
    "patch_application.py",
    "--module",
    module_path,
    "--description",
    description,
]
patch_application.main()
