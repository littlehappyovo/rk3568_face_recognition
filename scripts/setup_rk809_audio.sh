#!/usr/bin/env bash
# Enable the board speaker path without relying on a desktop PulseAudio session.
set -euo pipefail
audio_card="${FACEAPP_AUDIO_CARD:-rockchiprk809}"
audio_volume="${FACEAPP_AUDIO_VOLUME:-90%}"
amixer -q -c "$audio_card" sset 'Playback Path' SPK
amixer -q -c "$audio_card" sset 'Speaker' on
# The optional external-amplifier GPIO can read off when not wired; the codec's
# internal class-D output is selected by Playback Path=SPK.
amixer -q -c "$audio_card" sset 'DAC' "$audio_volume"
echo "RK809扬声器通路已启用，DAC音量=$audio_volume"
