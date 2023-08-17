from model import Model
from pytorch_lightning import Trainer
from data import Data

data = Data()
model = Model()

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    strategy="deepspeed_stage_2",
    precision="16-mixed"
)
trainer.fit(model, data)