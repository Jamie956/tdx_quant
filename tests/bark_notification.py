import requests
import os

key = os.getenv("BARK_KEY", "")
title = "脚本提醒"
content = "任务执行完成！"
url = f"https://api.day.app/{key}/{title}/{content}"
requests.get(url)
