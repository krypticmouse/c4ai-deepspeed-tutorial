from model import Model
from pytorch_lightning import Trainer
from data import Data

data = Data()
model = Model()

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    strategy="fsdp_cpu_offload"
)
trainer.fit(model, data)