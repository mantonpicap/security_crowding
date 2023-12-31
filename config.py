import os
from datetime import date
from rework_backtrader.utils.directory_controller import create_if_not_existed

to_time = (date.today()).strftime("%Y-%m-%d")

DATA_PARENT_FOLDER = f"{os.getenv('DATA_PARENT_FOLDER', os.getcwd())}"
DATA_FOLDER = os.path.join(DATA_PARENT_FOLDER, 'data_folder')
create_if_not_existed(DATA_FOLDER)
DAILY_DATA_FOLDER = os.path.join(DATA_FOLDER, f'{to_time}')
create_if_not_existed(DAILY_DATA_FOLDER)
