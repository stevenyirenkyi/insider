import pandas as pd
import warnings
import os

from sdv.tabular import CTGAN, CopulaGAN, GaussianCopula, TVAE
from utils import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger("general", "generative_models.log")


generative_models = {
    "GaussianCopula": lambda: GaussianCopula(),
    "CopulaGAN": lambda: CopulaGAN(),
    "CTGAN": lambda: CTGAN(),
    "TVAE": lambda: TVAE()
}


def get_malicious_samples(data: pd.DataFrame):
    removed_cols = ['user', 'day', 'week', 'starttime',
                    'endtime', 'sessionid', 'insider', 'timeind', 'Unnamed: 0']
    x_cols = [i for i in data.columns if i not in removed_cols]

    first_half = data[data.week <= max(data.week)/2]
    return first_half[first_half.insider > 0][x_cols]


data = pd.read_csv("./dataset/weekr5.2.csv")
malicious_samples = get_malicious_samples(data)


for name, model_factory in generative_models.items():
    model = model_factory()

    log.info(f"{name} start")
    model.fit(malicious_samples)
    log.info(f"{name} end")

    if not os.path.exists("./generative_models/"):
        os.makedirs("./generative_models/")
    model.save(f"./generative_models/{name.lower()}.pkl")
