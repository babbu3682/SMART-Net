from create_datasets.Hemorrhage import *
    


def build_dataset(is_train, args):
    mode='train' if is_train else 'valid'

    if args.data_set == 'Hemo_Uptask':
        dataset, collate_fn = Hemo_Uptask_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    elif args.data_set == 'Hemo_Downtask':
        dataset, collate_fn = Hemo_Downtask_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn



def build_test_dataset(args):
    dataset, collate_fn = Hemo_TEST_Dataset(test_dataset_name=args.test_dataset_name, data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn
