from datasets.Hemorrhage import *
    
def build_dataset(is_train, args):
    
    mode='train' if is_train else 'valid'

    if args.data_set == 'Hemo_Uptask':
        dataset, collate_fn = Hemo_Uptask_Dataset(mode=mode)

    elif args.data_set == 'Hemo_Uptask_Recon':
        dataset, collate_fn = Hemo_Uptask_Recon_Dataset(mode=mode, recon_option=args.recon_option)

    elif args.data_set == 'Hemo_Downtask':
        dataset, collate_fn = Hemo_Downtask_Dataset(mode=mode)

    return dataset, collate_fn




def build_test_dataset(args):
    dataset, collate_fn = Hemo_TEST_Dataset(test_data_name=args.test_data_name, recon_option=args.recon_option)

    return dataset, collate_fn

