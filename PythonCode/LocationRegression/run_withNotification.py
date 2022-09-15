import os
import requests

try:
    os.system('python3 Batch_pytorch_distance.py')
    requests.get(
        'https://api.telegram.org/bot5269105245:AAGCdJAZ9fzfazxC8nc-WI6MTSrxn2QC52U/sendMessage?chat_id=5520161508&text=Done')
except:
    requests.get(
        'https://api.telegram.org/bot5269105245:AAGCdJAZ9fzfazxC8nc-WI6MTSrxn2QC52U/sendMessage?chat_id=5520161508&text=Error')
