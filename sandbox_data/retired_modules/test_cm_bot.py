from sandbox.preseed_bootstrap import initialize_bootstrap_context
import menace_sandbox.capital_management_bot as cm

ctx = initialize_bootstrap_context("CapitalManagementBot")
cm._pipeline_instance = ctx["pipeline"]  # optional: short-circuit pipeline bootstrap
bot = cm.CapitalManagementBot(manager=ctx["manager"])

bot.apply_capital_constraints()
