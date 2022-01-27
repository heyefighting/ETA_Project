import json
import torch
import logger

from models.DeepTravel import DeepTravel

import utils
import data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_set, eval_set, dt_logger, file_name):
    if torch.cuda.is_available():
        model.cuda()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1.91981e-3)  # 使用Adam优化器 (输入为网络的参数和学习率)1e-3

    num_of_epochs = 100

    for epoch in range(0, num_of_epochs):

        for data_file in train_set:

            model.train()
            # print('train')
            data_iter = data_loader.get_loader(data_file, 1)

            recorder = open(file_name, 'a+')
            running_loss = 0.0
            for idx, (stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers) in enumerate(data_iter):
                # print('dataloading')
                # print(stats, temporal, spatial, dr_state, short_ttf, '\n\n',long_ttf,  '\n\n',helpers)
                stats, temporal, spatial, dr_state = utils.to_var(stats), utils.to_var(temporal), utils.to_var(
                    spatial), utils.to_var(dr_state)
                short_ttf, long_ttf = utils.to_var(short_ttf), utils.to_var(long_ttf)

                loss = model.evaluate(stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers)
                optimizer.zero_grad()
                sum = loss.sum()
                sum.backward()
                optimizer.step()

                running_loss += loss.mean().data.item()
                # print(loss.mean().data.item())
                # print(running_loss)
            recorder.write(str(running_loss) + "\n")
            recorder.close()
            print(running_loss)
        torch.save(model, "model_files/model_after_" + str(epoch) + "_epoch.pth")


def predict(model, eval_set, dt_logger):
    for data_file in eval_set:
        data_iter = data_loader.get_loader(data_file, 1)
        running_loss = 0.0
        for idx, (stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers) in enumerate(data_iter):
            stats, temporal, spatial, dr_state = utils.to_var(stats), utils.to_var(temporal), utils.to_var(
                spatial), utils.to_var(dr_state)
            short_ttf, long_ttf = utils.to_var(short_ttf), utils.to_var(long_ttf)

            loss = model.evaluate(stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers)
            running_loss += loss.mean().data.item()
        print(running_loss)


def main():
    file_name = "loss_recorder.txt"

    config = json.load(open('./config.json', 'r'))  # 将json文件打开然后就直接读取
    dt_logger = logger.get_logger()  # 日志信息设置

    model = DeepTravel()  # 这时候DeepTravel类的__init__这个函数会被调用
    # model = torch.load("./model_files/model_after_34_epoch.pth")
    train(model, config['train_set'], config['eval_set'], dt_logger, file_name)
    # predict(model, config['eval_set'], dt_logger)


if __name__ == '__main__':
    main()
