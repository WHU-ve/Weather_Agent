#!/bin/bash

# Usage:
#   source scripts/apply_runtime_profile.sh [fast|balanced|quality]

WEATHER_AGENT_PROFILE="${1:-balanced}"

case "$WEATHER_AGENT_PROFILE" in
  fast)
    export ENABLE_DYNAMIC_REPLAN=1
    export DYNAMIC_REPLAN_MAX_STEPS=2
    export DYNAMIC_REPLAN_PERCEPTION_INTERVAL=2
    export DYNAMIC_REPLAN_MIN_IMPROVE=0.0
    export TASK_THRESHOLD=0.72
    export TASK_MIN_TRIGGER_SCORE=0.55
    export TASK_MAX_REPEAT_PER_STEP=1
    export ENABLE_STEP_SCORE_GUARD=1
    export STEP_SCORE_MAX_DROP=0.01
    ;;
  balanced)
    export ENABLE_DYNAMIC_REPLAN=1
    export DYNAMIC_REPLAN_MAX_STEPS=4
    export DYNAMIC_REPLAN_PERCEPTION_INTERVAL=1
    export DYNAMIC_REPLAN_MIN_IMPROVE=0.0
    export TASK_THRESHOLD=0.65
    export TASK_MIN_TRIGGER_SCORE=0.45
    export TASK_MAX_REPEAT_PER_STEP=2
    export ENABLE_STEP_SCORE_GUARD=1
    export STEP_SCORE_MAX_DROP=0.005
    ;;
  quality)
    export ENABLE_DYNAMIC_REPLAN=1
    export DYNAMIC_REPLAN_MAX_STEPS=6
    export DYNAMIC_REPLAN_PERCEPTION_INTERVAL=1
    export DYNAMIC_REPLAN_MIN_IMPROVE=0.001
    export TASK_THRESHOLD=0.58
    export TASK_MIN_TRIGGER_SCORE=0.35
    export TASK_MAX_REPEAT_PER_STEP=2
    export ENABLE_STEP_SCORE_GUARD=1
    export STEP_SCORE_MAX_DROP=0.0
    ;;
  *)
    echo "Unknown profile: $WEATHER_AGENT_PROFILE"
    echo "Use one of: fast | balanced | quality"
    return 1 2>/dev/null || exit 1
    ;;
esac

export WEATHER_AGENT_PROFILE
echo "[PROFILE] runtime profile: $WEATHER_AGENT_PROFILE"