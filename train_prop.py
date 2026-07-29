import csv
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import time
from typing import Any, Dict, Union, Tuple
import pickle as pk
import os
import torch.distributed as dist

import torch
from jarvis.core.atoms import Atoms

from jarvis.db.jsonutils import dumpjson, loadjson
from torch import nn

import ignite
from tqdm import tqdm

from data import get_train_val_loaders

from models.config import TrainingConfig
import json
import pprint
import math
import torch
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers.stores import EpochOutputStore
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# from ignite.engine import (
#     Events,
#     Engine,
#     # create_supervised_evaluator,
#     # create_supervised_trainer,
# )
from models.bdipnet import BDIPNet

import random

plt.switch_backend("agg")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda")


def prepare_batch(
        batch, device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    batch = (
        batch.to(device, non_blocking=non_blocking),
        batch.y.to(device, non_blocking=non_blocking),
    )

    return batch


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total Trainable Params: {total_params}")
    return total_params


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer

class Events(Enum):
    ITERATION_COMPLETED = "iteration_completed"
    EPOCH_COMPLETED = "epoch_completed"


class TrainerState:
    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.output = None
        self.should_terminate = False


class SupervisedTrainer:
    def __init__(
        self,
        net,
        optimizer,
        criterion,
        prepare_batch,
        device,
        scheduler=None,
        deterministic=False,
    ):
        self.net = net.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.prepare_batch = prepare_batch
        self.device = device
        self.scheduler = scheduler

        self.state = TrainerState()
        self._event_handlers = defaultdict(list)

        if deterministic:
            torch.use_deterministic_algorithms(True)

    def add_event_handler(self, event, handler):
        self._event_handlers[event].append(handler)
        return handler

    def _fire_event(self, event):
        for handler in self._event_handlers[event]:
            handler(self)

    def terminate(self):
        self.state.should_terminate = True

    def run(self, train_loader, max_epochs=1):
        self.state = TrainerState()

        for epoch in range(max_epochs):
            self.state.epoch = epoch + 1
            self.net.train()

            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                self.state.iteration += 1

                x, y = self.prepare_batch(
                    batch,
                    device=self.device,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(set_to_none=True)

                y_pred = self.net(x)
                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                batch_loss = loss.detach().item()
                self.state.output = batch_loss

                total_loss += batch_loss
                num_batches += 1

                self._fire_event(Events.ITERATION_COMPLETED)

                if self.state.should_terminate:
                    break

            epoch_loss = total_loss / max(num_batches, 1)
            self.state.output = epoch_loss

     
     
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()

            self._fire_event(Events.EPOCH_COMPLETED)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{self.state.epoch}/{max_epochs}] "
                f"- Loss: {epoch_loss:.6f} "
                f"- LR: {current_lr:.8f}"
            )

            if self.state.should_terminate:
                break

        return self

def create_supervised_trainer(
    net,
    optimizer,
    criterion,
    prepare_batch,
    device,
    scheduler=None,
    deterministic=False,
):
    return SupervisedTrainer(
        net=net,
        optimizer=optimizer,
        criterion=criterion,
        prepare_batch=prepare_batch,
        device=device,
        scheduler=scheduler,
        deterministic=deterministic,
    )
def train_pyg(
        config: Union[TrainingConfig, Dict[str, Any]],
        data_root: str = None,
        file_format: str = 'poscar',
        checkpoint: str = None,
        testing: bool = False,
        train_val_test_loaders=None,
        fold: int = None,
        db: str = "bidb"
):
   
    config = TrainingConfig(**config)
    if config.random_seed is not None:
            deterministic = True
            ignite.utils.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            torch.cuda.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)
            random.seed(config.random_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    base_output_dir = config.output_dir
    fold_output_dir = os.path.join(base_output_dir, f"{db}_fold_{fold}")
    fold_cache_dir = os.path.join(fold_output_dir, config.cache_dir)
    fold_process_dir = os.path.join(fold_output_dir, config.process_dir)

    os.makedirs(fold_output_dir, exist_ok=True)
    os.makedirs(fold_cache_dir, exist_ok=True)
    os.makedirs(fold_process_dir, exist_ok=True)

    checkpoint_dir = os.path.join(
        fold_output_dir,
        config.checkpoint_dir,
    )


    print("config:")
    tmp = config.dict()

    with open(
        os.path.join(fold_output_dir, "config.json"),
        "w",
    ) as f:
        json.dump(tmp, f, indent=4)
    

    if data_root:
        dataset_info = loadjson(f"dataset_{db}_info_{fold}.json")
        if "n_train" in dataset_info:
            config.n_train = dataset_info['n_train']
        if "n_val" in dataset_info:
            config.n_val = dataset_info['n_val']
        if "n_test" in dataset_info:
            config.n_test = dataset_info['n_test']

        if "train_ratio" in dataset_info:
            config.train_ratio = dataset_info['train_ratio']
        if "val_ratio" in dataset_info:
            config.val_ratio = dataset_info['val_ratio']
        if "test_ratio" in dataset_info:
            config.test_ratio = dataset_info['test_ratio']

        config.keep_data_order = True
        config.target = "target"

        #id_prop_dat = os.path.join(f"data/reg-bidb-stack-vasp/id_prop_{db}_fold_{fold}.csv")
        #id_prop_dat = os.path.join(f"data/reg-hetdb-stack-vasp/id_prop_all_mono_fold_{fold}.csv")
        id_prop_dat = os.path.join(f"data/reg-samba-stack-vasp/id_prop_{db}_fold_{fold}.csv")
       
        with open(id_prop_dat, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        file_format = "poscar"
        dataset_array = []
        for i in data:
            info = {}
            file_name = i[0]
            file_path = os.path.join(data_root, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                # Note using 500 angstrom as box size
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                # Note using 500 angstrom as box size
                # Recommended install pytraj
                # conda install -c ambermd pytraj
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
            
            z_cut = float(first_line.split()[0])
            #print(f"z_cut: {z_cut} for file: {file_name}")

            atoms_dict = atoms.to_dict()
            atoms_dict["z_cut"] = z_cut
            info["atoms"] = atoms_dict
            info["jid"] = file_name
            info["z_cut"] = z_cut

            info["target"] = float(i[1])
            dataset_array.append(info)
    else:
        dataset_array = None
    
    print('output_dir train', config.output_dir)
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        (
            train_loader,
            val_loader,
            test_loader,
            mean,
            std
        ) = get_train_val_loaders(
            dataset=config.dataset,
            root=fold_output_dir,
            cachedir=fold_cache_dir,
            processdir=fold_process_dir,
            dataset_array=dataset_array,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            normalize=config.normalize,
            euclidean=config.euclidean,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            infinite_funcs=config.infinite_funcs,
            infinite_params=config.infinite_params,
            R=config.R,
            keep_data_order=config.keep_data_order,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        mean = 0.0
        std = 1.0

    # define network, optimizer, scheduler
    _model = {
        "bdipnet": BDIPNet,
    }
    config.model.euclidean = config.euclidean
    net = _model.get(config.model.name)(config.model)
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint)["model"])

    count_parameters(net)
    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    config.scheduler = "step"
    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=200, gamma=0.25
        )

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[config.criterion]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError() * std, "neg_mae": -1.0 * MeanAbsoluteError() * std}

    trainer = create_supervised_trainer(
    net,
    optimizer,
    criterion,
    prepare_batch=prepare_batch,
    device=device,
    scheduler=scheduler,
    deterministic=deterministic,
)

    # trainer.add_event_handler(
    #     Events.EPOCH_COMPLETED,
    #     TerminateOnNan(),
    # )

    trainer.run(
        train_loader,
        max_epochs=config.epochs,
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        # handler = Checkpoint(
        #     to_save,
        #     DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        #     n_saved=5,
        #     score_name="neg_mae",
        #     global_step_transform=lambda *_: trainer.state.epoch,
        # )
        #evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    # if config.progress:
    #     pbar = ProgressBar()
    #     pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # history = {
    #     "train": {m: [] for m in metrics.keys()},
    #     "validation": {m: [] for m in metrics.keys()},
    # }

    if config.store_outputs:
        eos = EpochOutputStore()
        #eos.attach(evaluator)
        train_eos = EpochOutputStore()
        #train_eos.attach(train_evaluator)

    # collect evaluation performance
    #@trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        #train_evaluator.run(train_loader)
        #evaluator.run(val_loader)

        #tmetrics = train_evaluator.state.metrics
        #vmetrics = evaluator.state.metrics
        #for metric in metrics.keys():
            #tm = tmetrics[metric]
            #vm = vmetrics[metric]
            #if isinstance(tm, torch.Tensor):
            #    tm = tm.cpu().numpy().tolist()
                #vm = vm.cpu().numpy().tolist()

            #history["train"][metric].append(tm)
            #history["validation"][metric].append(vm)

        # if config.store_outputs:
        #     history["EOS"] = eos.data
        #     history["trainEOS"] = train_eos.data
        #     dumpjson(
        #         filename=os.path.join(config.output_dir, config.model.name + "_" + config.target + "_history_val.json"),
        #         data=history["validation"],
        #     )
        #     dumpjson(
        #         filename=os.path.join(config.output_dir,
        #                               config.model.name + "_" + config.target + "_history_train.json"),
        #         data=history["train"],
        #     )
        config.progress=True
        if config.progress:
            pbar = ProgressBar()

            current_lr = optimizer.param_groups[0]["lr"]

            pbar.log_message(
                f"Epoch {engine.state.epoch}/{config.epochs} "
                f"| LR: {current_lr:.8f}"
            )

    # if config.n_early_stopping is not None:
    #     def default_score_fn(engine):
    #         score = engine.state.metrics["mae"]
    #         return score

    #     es_handler = EarlyStopping(
    #         patience=config.n_early_stopping,
    #         score_function=default_score_fn,
    #         trainer=trainer,
    #     )
    #     evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

    # if config.log_tensorboard:

    #     tb_logger = TensorboardLogger(
    #         log_dir=os.path.join(config.output_dir, "tb_logs", "test")
    #     )
    #     for tag, evaluator in [
    #         ("training", train_evaluator),
    #         ("validation", evaluator),
    #     ]:
    #         tb_logger.attach_output_handler(
    #             evaluator,
    #             event_name=Events.EPOCH_COMPLETED,
    #             tag=tag,
    #             metric_names=["loss", "mae", "neg_mae"],
    #             global_step_transform=global_step_from_engine(trainer),
    #         )

    # train the model!
    # if not testing:
    #     trainer.run(train_loader, max_epochs=config.epochs)

    # if config.log_tensorboard:
    #     test_loss = evaluator.state.metrics["loss"]
    #     tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
    #     tb_logger.close()

    print("Testing!")
    net.eval()
    t1 = time.time()

    prediction_file = None
    config.write_predictions=True
    if config.write_predictions:
        prediction_file = open(
            os.path.join(f"results/{db}_{fold}_test_0.csv"),
            "w",
        )
        prediction_file.write("id,target,prediction\n")

    targets = []
    predictions = []

    #Lấy mean/std về scalar
    if isinstance(mean, torch.Tensor):
        mean_value = float(mean.detach().cpu().reshape(-1)[0].item())
    else:
        mean_value = float(np.asarray(mean).reshape(-1)[0])

    if isinstance(std, torch.Tensor):
        std_value = float(std.detach().cpu().reshape(-1)[0].item())
    else:
        std_value = float(np.asarray(std).reshape(-1)[0])

    all_ids = list(test_loader.dataset.ids)
    id_index = 0

    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            # Prediction của toàn bộ batch
            batch_predictions = net(data.to(device))

            batch_predictions = (
                batch_predictions
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            # Target của toàn bộ batch
            batch_targets = (
                data.label
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            mean_value = 0.0
            std_value = 1.0
            # Denormalize predictions
            batch_predictions = (
                batch_predictions * std_value + mean_value
            )

            if len(batch_predictions) != len(batch_targets):
                raise ValueError(
                    "Prediction/target size mismatch: "
                    f"{len(batch_predictions)} predictions vs "
                    f"{len(batch_targets)} targets."
                )

            batch_size = len(batch_targets)
            batch_ids = all_ids[id_index:id_index + batch_size]
            id_index += batch_size

            # extend thay vì append để list chỉ chứa scalar
            targets.extend(batch_targets.astype(float).tolist())
            predictions.extend(
                batch_predictions.astype(float).tolist()
            )

            if prediction_file is not None:
                for sample_id, target_value, prediction_value in zip(
                    batch_ids,
                    batch_targets,
                    batch_predictions,
                ):
                    prediction_file.write(
                        f"{sample_id},"
                        f"{float(target_value):.6f},"
                        f"{float(prediction_value):.6f}\n"
                    )

    if prediction_file is not None:
        prediction_file.close()

    t2 = time.time()
    #print("Test time(s):", t2 - t1)

    from sklearn.metrics import mean_absolute_error

    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    print("targets shape:", targets.shape)
    print("predictions shape:", predictions.shape)

    test_mae = mean_absolute_error(targets, predictions)

    print("Test MAE:", test_mae)

    return test_mae


def train_prop_model(config: Dict, data_root: str = None, checkpoint: str = None, testing: bool = False, file_format: str = 'poscar',fold: int = 0,db="bidb"):
    if config["dataset"] == "megnet":
        config["id_tag"] = "id"
        if config["target"] == "e_form" or config["target"] == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239

    result = train_pyg(config, data_root=data_root, file_format=file_format, checkpoint=checkpoint, testing=testing, fold=fold,db=db)
    return result