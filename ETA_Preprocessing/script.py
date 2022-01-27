import collections
import helpers  # 自己写的helpers
import argparse
import logger  # 自己写的logger
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=int, default=256)
parser.add_argument('--ttf_destination_folder', type=str, default='../traffic_features/')
parser.add_argument('--data_destination_folder', type=str, default='../processed_data/')

args = parser.parse_args()


def define_travel_grid_path(data, coords, short_ttf, long_ttf, n):
    """
    coords=[30.09, 102.9, 31.44, 104.9]  # 路网的西南、东北点的纬、经度
    n=256
    """

    # compute size of grid cell：返回cell_params:一个方格跨过多少经、纬度
    # *coords:将coords分为四个独立的参数，n=256
    cell_params = helpers.define_grid_cell(*coords, n)      # 切割路网，形成一个个相邻的矩形小区域

    for dd in data:
        # relative coordinates (as first cell has coordinates coords[1][0]
        # dd['key']:字典dd按key访问元素
        x = np.array(dd['lngs']) - coords[1]  # 经度
        y = np.array(dd['lats']) - coords[0]  # 纬度

        # T_path - sequence of grid indices that correspond historical gps points
        # G_path - sequence of grid indices  of full path (with intermediate cells without gps points)
        dd['T_X'], dd['T_Y'], dd['G_X'], dd['G_Y'], dd['hour_bin'], dd['time_bin'], dd['dr_state'], dd['borders'], dd[
            'mask'] = helpers.map_gps_to_grid(
            x, y,
            dd['timeID'], dd['weekID'],
            dd['time_gap'], dd['dist_gap'],
            cell_params,
            short_ttf, long_ttf,
            dd['dist']
        )

        dd['day_bin'] = [dd['weekID'] for _ in dd['G_X']]


def main():
    config = helpers.read_config()
    e_logger = logger.get_logger()

    # initialize arrays for short-term and long-term traffic features
    speed_array = 'speeds'
    time_array = 'times'
    # 使用collections.defaultdict()方法来为字典提供默认值,256*256
    # lambda:匿名函数，没有函数名的函数
    short_ttf = [
        [collections.defaultdict(lambda: {speed_array: [], time_array: []}) for _ in range(256)] for _ in range(256)
    ]
    long_ttf = [
        [collections.defaultdict(lambda: {speed_array: [], time_array: []}) for _ in range(256)] for _ in range(256)
    ]

    for data_file in config['data']:  # 从"train_00", "train_01", "train_02", "train_03", "train_04"文件中
        e_logger.info('Generating G and T paths and extracting traffic features on {} ...'.format(data_file))

        data = helpers.read_data(data_file)
        # print(data)
        # config["coords"]: [30.09, 102.9, 31.44, 104.9],args.grid_size=256
        define_travel_grid_path(data, config['coords'], short_ttf, long_ttf, args.grid_size)

        e_logger.info(
            'Saving extended with G and T paths data in {}{}.\n'.format(args.data_destination_folder, data_file))

        # args.data_destination_folder：'../processed_data/'
        helpers.save_processed_data(data, args.data_destination_folder, data_file)

    e_logger.info('Aggregate historical traffic features ...')
    helpers.aggregate_historical_data(short_ttf, long_ttf)
    e_logger.info('Saving extracted traffic features in {}'.format(args.ttf_destination_folder))
    helpers.save_extracted_traffic_features(short_ttf, long_ttf, args.ttf_destination_folder)


if __name__ == '__main__':
    main()
