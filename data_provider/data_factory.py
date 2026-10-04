from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute


data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
}


def data_provider(args, flag):
    data_class = data_dict[args.data]
    dataset = data_class(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=not getattr(args, "disable_scale", False),
        timeenc=0 if args.embed != "timeF" else 1,
        freq=args.freq,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag != "test",
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, loader
