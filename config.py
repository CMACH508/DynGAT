params = {
    "target_dataset": 'bitcoin-elliptic',
    "epochs": 3,
    "learning_rate": 1e-5,
    "weight_decay": 5e-4,
    "early_stop_thres": 1000,
    "eval_after_epochs": 1,
    "save_span": -1,
    "task": 'test',
    "gpu": 0,

    "num_gnn_layers": 2,
    "hidden_n_size": 64,
    "hidden_e_size": 2,
    "num_heads": 1,
    "attn_dropout": 0.0,
    "time_cuts": 5,
    "t_dim": 3,
    'loss_weight': [0.7, 0.3],

    "experiment_name": 'test1',
    "log_path": './save',
    "log_on_console": False,
    "log_interval": 0
}