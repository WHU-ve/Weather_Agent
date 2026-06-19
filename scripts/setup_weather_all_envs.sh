#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/setup_weather_env.sh "${1:-weather_agent}"
bash scripts/setup_weather_ridcp_env.sh "${2:-weather_agent_ridcp}"
bash scripts/setup_weather_nafnet_env.sh "${3:-weather_agent_nafnet}"
bash scripts/setup_weather_maxim_env.sh "${4:-weather_agent_maxim}"
bash scripts/setup_weather_diffplugin_env.sh "${5:-weather_agent_diffplugin}"
bash scripts/setup_weather_jstasr_env.sh "${6:-weather_agent_jstasr}"

echo "All weather-agent environments are prepared."
