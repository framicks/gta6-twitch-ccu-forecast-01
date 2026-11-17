# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/framicks/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/framicks/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast of Grand Theft Auto VI's launch-day Twitch viewers, hour by hour, for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty. `main.py` (Python 3.13, standard library only) learns the intraday shape of a big console launch from the case's comparables: per hour, the mean log-ratio of viewers to the hour-1 print across past launches whose hour 1 was already live, with an IQR-based spread for the quantile widths. The learned curve dips in the deep night and rises to the evening peak — the shape the flat house baseline ignores. Local validation: practice cases 0.308 vs house baseline 0.387 (pinball loss, lower is better); leave-one-out over the comparables 0.238 vs 0.284.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=erdians&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
