import os
from utils import fetch_data_for_fine_tuning, organize_data
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='MURA data organization')
    parser.add_argument('--data-in', help='input data location', default="MURA-v1.1",
                        type=str)
    parser.add_argument('--data-out', help='output data location',
                        default="", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    if args.data_out == "":
        data_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    else:
        data_out = args.data_out

    data100 = os.path.join(data_out, "processed_100")
    print("Starting data organization...")
    organize_data(args.data_in, data100)
    print("100% split complete!")
    fetch_data_for_fine_tuning(0.01, verify_study_type_balance=True, input_pth=data100,
                               output_pth=os.path.join(data_out, "processed_1"))
    print("1% split complete!")
    fetch_data_for_fine_tuning(0.1, verify_study_type_balance=True, input_pth=data100,
                               output_pth=os.path.join(data_out, "processed_10"))
    print("10% split complete!")

