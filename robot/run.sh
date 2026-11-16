#!/usr/bin/env bash
# Start the study system on the PC that controls MiRo-E. Needs the MiRo Developer Kit (MDK) and the ROS master on
# the robot. The VLM key is read from the environment and must never be written into a file in this repository:
#   export GEMINI_API_KEY=...
# Usage:
#   MIRO_IP=<robot ip> PC_IP=<this pc ip> bash run.sh director [organic|mechanical]   # terminal 1, keep running
#   MIRO_IP=<robot ip> PC_IP=<this pc ip> bash run.sh ui                               # terminal 2: S / M / X keys
# Before each participant: python3 profile_loader.py <code> (writes tuner_profile.json from the Tuner export).
set -euo pipefail
: "${MIRO_IP:?set MIRO_IP}" "${PC_IP:?set PC_IP}"
source ~/mdk/setup.bash
export MIRO_ROBOT_NAME=miro
export ROS_MASTER_URI="http://$MIRO_IP:11311"
export ROS_IP="$PC_IP"
cd "$(dirname "$0")"
case "${1:-}" in
  director) python3 miro_director.py _noise_mode:="${2:-organic}" ;;
  ui)       : "${GEMINI_API_KEY:?set GEMINI_API_KEY (needed by the Spontaneous pipeline)}"
            python3 experiment_ui.py ;;
  *)        echo "usage: run.sh director [organic|mechanical] | ui"; exit 1 ;;
esac
