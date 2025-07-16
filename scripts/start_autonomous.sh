#!/bin/sh
python -c 'import auto_env_setup, run_autonomous, sys; auto_env_setup.ensure_env(); run_autonomous.main(sys.argv[1:])' "$@"
