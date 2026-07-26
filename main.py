from train_prop import train_prop_model

import yaml
import argparse

# Set up argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training configuration of PotNet')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to the output')
    parser.add_argument('--data_root', type=str, default=None, help='Path to the data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--testing', action='store_true', help='Evaluation phase')
    parser.add_argument('--fold', type=int, help='k_fold')

    # Parse arguments
    args = parser.parse_args()
    
    # Load data from YAML file
    with open(args.config, 'r') as file:
        data = yaml.safe_load(file)

    
    #args.data_root=f"data/reg-bidb-stack-vasp"
    #args.data_root=f"data/reg-hetdb-stack-vasp"
    args.data_root=f"data/reg-samba-stack-vasp"


    #db="bidb_stack"
    #db="hetdb_stack"
    db="samba_full_stack_layer"
  



    data["output_dir"] = args.output_dir
    train_prop_model(data, data_root=args.data_root, checkpoint=args.checkpoint, testing=args.testing,fold=args.fold,db=db)