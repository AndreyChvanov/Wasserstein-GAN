import neptune.new as neptune
from neptune.new.types import File


class NeptuneLogger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.initialize()

    def initialize(self):
        self.run = neptune.init_run(self.cfg.project_name,
                                api_token=self.cfg.api_token,
                                )

    def end_logging(self):
        self.run.stop()

    def add_scalar(self, name, value, step):
        self.run[name].log(value, step=step)

    def log_metrics(self, names, metrics, step):
        for name, metric in zip(names, metrics):
            self.run[name].log(metric, step=step)
            # neptune.log_metric(name, metric)

    def log_configs(self, names, cfgs):
        for name, cfg in zip(names, cfgs):
            self.run[f'configs/{name}'].log(cfg)

    def log_images(self, names, images):
        for name, img in zip(names, images):
            self.run[name].upload(File(img))