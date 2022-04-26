from create_datasets.Hemorrhage import *
    


def build_dataset(is_train, args):
    mode='train' if is_train else 'valid'

    if args.training_stream == 'Upstream':
        dataset, collate_fn = Hemo_Uptask_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    elif args.training_stream == 'Downstream':
        dataset, collate_fn = Hemo_Downtask_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn

def build_dataset_imbalance(mode, args):

    if args.training_stream == 'Upstream':
        dataset, collate_fn = Hemo_Uptask_Imbalance_Dataset(mode=mode, data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn



def build_test_dataset(args):
    
    if args.slice_wise_manner:
        dataset, collate_fn = Hemo_TEST_Dataset_Slicewise(test_dataset_name=args.test_dataset_name, data_folder_dir=args.data_folder_dir)

    else :
        dataset, collate_fn = Hemo_TEST_Dataset(test_dataset_name=args.test_dataset_name, data_folder_dir=args.data_folder_dir)

    return dataset, collate_fn

