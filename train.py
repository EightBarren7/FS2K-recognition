from tqdm import tqdm
from model import get_model
from dataloader import dataloader
import pandas as pd
from parameters import *

train_iter = dataloader(train_json_path, data_dir, batch_size, True)
def train_v1(model_name):
    train_iter = dataloader(train_json_path, data_dir, batch_size, True)

    model = get_model(model_name, 'v1')
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss1_list = []
    loss2_list = []
    loss3_list = []
    loss4_list = []
    loss5_list = []
    loss6_list = []
    loss_list = []

    for epoch in range(num_epochs):
        with tqdm(train_iter, unit='batch') as tepoch:
            total_loss = 0.
            for data, label in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1} train")
                data = data.to(device)
                label = label.to(device)
                outputs = model(data)
                loss1 = loss_fn(outputs[0], label[:, 0])
                loss1_list.append(loss1.item())
                loss2 = loss_fn(outputs[1], label[:, 1])
                loss2_list.append(loss2.item())
                loss3 = loss_fn(outputs[2], label[:, 2])
                loss3_list.append(loss3.item())
                loss4 = loss_fn(outputs[3], label[:, 3])
                loss4_list.append(loss4.item())
                loss5 = loss_fn(outputs[4], label[:, 4])
                loss5_list.append(loss5.item())
                loss6 = loss_fn(outputs[5], label[:, 5])
                loss6_list.append(loss6.item())
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                loss_list.append(loss.item())
                total_loss += loss.item() / len(train_iter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=total_loss)

    torch.save(model.state_dict(), './result/model/{}.pth'.format(
        model_name+'_epochs_'+str(num_epochs)+'_lr_'+str(lr)+'_v1'
    ))

    out_dict = {
        'hair': loss1_list,
        'hair_color': loss2_list,
        'gender':  loss3_list,
        'earring': loss4_list,
        'smile': loss5_list,
        'frontal_face': loss6_list,
        'total_loss': loss_list
    }
    out = pd.DataFrame(out_dict)
    out.to_csv('./result/loss/{}.csv'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr) + '_v1'
    ), index=False)


def train_v2(model_name):
    train_iter = dataloader(train_json_path, data_dir, batch_size, True)

    model = get_model(model_name, 'v2')
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss1_list = []
    loss2_list = []
    loss3_list = []
    loss4_list = []
    loss5_list = []
    loss_list = []

    for epoch in range(num_epochs):
        with tqdm(train_iter, unit='batch') as tepoch:
            total_loss = 0.
            for data, label in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1} train")
                data = data.to(device)
                label = label.to(device)
                outputs = model(data)
                loss1 = loss_fn(outputs[0], label[:, 0])
                loss1_list.append(loss1.item())
                loss2 = loss_fn(outputs[1], label[:, 2])
                loss2_list.append(loss2.item())
                loss3 = loss_fn(outputs[2], label[:, 3])
                loss3_list.append(loss3.item())
                loss4 = loss_fn(outputs[3], label[:, 4])
                loss4_list.append(loss4.item())
                loss5 = loss_fn(outputs[4], label[:, 5])
                loss5_list.append(loss5.item())
                loss = loss1 + loss2 + loss3 + loss4 + loss5 / 5
                loss_list.append(loss.item())
                total_loss += loss.item() / len(train_iter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=total_loss)

    torch.save(model.state_dict(), './result/model/{}.pth'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr)+'_v2'
    ))

    out_dict = {
        'hair': loss1_list,
        'gender': loss2_list,
        'earring': loss3_list,
        'smile': loss4_list,
        'frontal_face': loss5_list,
        'total_loss': loss_list
    }
    out = pd.DataFrame(out_dict)
    out.to_csv('./result/loss/{}.csv'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr) + '_v2'
    ), index=False)


def train(model_name, model_version):
    print("training on:{}".format(device))
    if model_version == 'v1':
        train_v1(model_name)
    elif model_version == 'v2':
        train_v2(model_name)

