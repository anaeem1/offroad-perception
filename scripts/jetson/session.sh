#!/usr/bin/env bash
# One unattended board session: the main replay and the lidar-dropout replay (hazard H-6). The board drops off
# Wi-Fi and needs a physical restart to come back, so nothing here depends on the connection: start it inside
# tmux and it runs to the end with no one attached. Clocks are locked (fan at max) only for this session;
# temperature and power are logged throughout and synced to disk every 10 s; on exit, even after an error,
# the clocks are restored and any service it stopped is started again. The previous boot's journal is saved
# first, to explain the last disconnect.
# Usage: session.sh <bundle_dir> <clip> <dropout_clip> <engines_dir> <out_dir> [--quiet-board]
#   --quiet-board  stop the find_me_loc container and nomad.service for the session (owner approval first)
# Writes <out_dir>/{main,dropout,session}/ and <out_dir>.tar.gz (engines excluded); <out_dir>/session/DONE
# (the exit status) is written last.
set -o pipefail
B="$(realpath "${1:?bundle dir}")"; CLIP="$(realpath "${2:?clip}")"; DROP="$(realpath -m "${3:?dropout clip}")"
ENG="$(realpath "${4:?engines dir}")"; OUT="$(realpath -m "${5:?out dir}")"; QUIET="${6:-}"
[ -e "$OUT" ] && { echo "$OUT exists: use a new out dir"; exit 1; }
S="$OUT/session"; mkdir -p "$S"
exec > >(tee -a "$S/session.log") 2>&1
STOPPED=(); PIDS=()

restore() {
  local rc=$?
  echo "== restore ($(date '+%F %T'), exit $rc)"
  [ -f "$S/clocks_before.conf" ] && sudo -n jetson_clocks --restore "$S/clocks_before.conf" && echo "clocks restored"
  for s in "${STOPPED[@]}"; do
    [ "$s" = find_me_loc ] && docker start find_me_loc >/dev/null && echo "find_me_loc started"
    [ "$s" = nomad ] && sudo -n systemctl start nomad.service && echo "nomad.service started"
  done
  kill "${PIDS[@]}" 2>/dev/null
  tar czf "$OUT.tar.gz" --exclude=engines -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
  ls -la "$OUT.tar.gz"
  echo "$rc" > "$S/DONE"; sync
}
trap restore EXIT
trap 'exit 130' INT TERM HUP

sudo -v || { echo "sudo needs a password: tmux attach and type it, or run sudo -v first"; exit 1; }
( while sleep 60; do sudo -n true; done ) & PIDS+=($!)
( while sleep 10; do sync; done ) & PIDS+=($!)

echo "== board ($(date '+%F %T'))"
uptime; free -h; df -h ~ | tail -1; nvpmodel -q 2>/dev/null | head -1
for z in /sys/class/thermal/thermal_zone*; do t=$(cat "$z/temp" 2>/dev/null) && echo "  $(cat "$z/type") $((t / 1000)) C"; done
for i in $(iw dev 2>/dev/null | awk '/Interface/ {print $2}'); do echo "  $i $(iw dev "$i" get power_save 2>/dev/null)"; done
docker ps --format '{{.Names}}' 2>/dev/null | sed 's/^/  container running: /'
echo "  nomad.service $(systemctl is-active nomad.service 2>/dev/null)"

echo "== previous boot (why the last session ended)"
sudo -n journalctl --list-boots --no-pager 2>/dev/null | tail -4
sudo -n journalctl -b -1 --no-pager -o short-iso 2>/dev/null \
  | grep -iE "throttl|over.?current|soctherm|thermal|overheat|oom|out of memory|wlan|wifi|wpa_supplicant|deauth|link is down|watchdog|panic|hung task" \
  | tail -150 > "$S/previous_boot_events.txt"
sudo -n journalctl -b -1 --no-pager -o short-iso -n 60 > "$S/previous_boot_last_lines.txt" 2>&1
echo "  $(wc -l < "$S/previous_boot_events.txt") matching events in previous_boot_events.txt; last lines in previous_boot_last_lines.txt"

if [ "$QUIET" = --quiet-board ]; then
  docker ps --format '{{.Names}}' | grep -qx find_me_loc && docker stop find_me_loc >/dev/null && STOPPED+=(find_me_loc) \
    && echo "find_me_loc stopped"
  systemctl is-active --quiet nomad.service && sudo -n systemctl stop nomad.service && STOPPED+=(nomad) \
    && echo "nomad.service stopped"
fi

stdbuf -oL tegrastats --interval 2000 > "$S/tegrastats.log" 2>&1 & PIDS+=($!)
echo "== clocks locked, fan at max"
sudo -n jetson_clocks --store "$S/clocks_before.conf" && sudo -n jetson_clocks && sudo -n jetson_clocks --fan \
  && sudo -n jetson_clocks --show > "$S/clocks_locked.txt" || echo "clocks NOT locked"

mkdir -p "$OUT/main" "$OUT/dropout"
cp -r "$ENG" "$OUT/main/engines"; cp -r "$ENG" "$OUT/dropout/engines"
echo "== main replay ($(date '+%T'))"
bash "$B/replay_on_jetson.sh" "$B/ws" "$CLIP" "$B/onnx" "$OUT/main" 00001 || { echo "main replay failed"; exit 1; }
if [ ! -d "$DROP" ]; then
  echo "== dropout clip"
  bash -c "source /opt/ros/humble/setup.bash && python3 '$B/make_dropout_clip.py' '$CLIP' '$DROP'" || exit 1
fi
echo "== dropout replay ($(date '+%T'))"
bash "$B/replay_on_jetson.sh" "$B/ws" "$DROP" "$B/onnx" "$OUT/dropout" 00001 || { echo "dropout replay failed"; exit 1; }
echo "== replays done ($(date '+%T'))"
