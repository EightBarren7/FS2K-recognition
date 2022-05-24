import pandas as pd
from dataloader import dataloader
from tqdm import tqdm
from model import get_model
from parameters import *


def test_v1(model_name):
    test_iter = dataloader(test_json_path, data_dir, batch_size, False)

    model = get_model(model_name, 'v1')
    model.load_state_dict(torch.load('./result/model/{}.pth'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr)+'_v1')
    ))
    model.to(device)
    model.eval()

    hair_list = []
    hair_color_list = []
    gender_list = []
    earring_list = []
    smile_list = []
    frontal_face_list = []

    with tqdm(test_iter, unit='batch') as tepoch:
        total_acc = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = 0.
        for data, label in tepoch:
            tepoch.set_description(f"test")
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)

            acc1 += (outputs[0].argmax(1) == label[:, 0]).sum().item() / len(test_iter) / batch_size
            for i in outputs[0].argmax(1):
                hair_list.append(i.item())

            acc2 += (outputs[1].argmax(1) == label[:, 1]).sum().item() / len(test_iter) / batch_size
            for i in outputs[1].argmax(1):
                hair_color_list.append(i.item())

            acc3 += (outputs[2].argmax(1) == label[:, 2]).sum().item() / len(test_iter) / batch_size
            for i in outputs[2].argmax(1):
                gender_list.append(i.item())

            acc4 += (outputs[3].argmax(1) == label[:, 3]).sum().item() / len(test_iter) / batch_size
            for i in outputs[3].argmax(1):
                earring_list.append(i.item())

            acc5 += (outputs[4].argmax(1) == label[:, 4]).sum().item() / len(test_iter) / batch_size
            for i in outputs[4].argmax(1):
                smile_list.append(i.item())

            acc6 += (outputs[5].argmax(1) == label[:, 5]).sum().item() / len(test_iter) / batch_size
            for i in outputs[5].argmax(1):
                frontal_face_list.append(i.item())

            total_acc = (acc1 + acc2 + acc3 + acc4 + acc5 + acc6) / 6
            tepoch.set_postfix(total_acc=total_acc)

    out_dict = {
        'hair': hair_list,
        'hair_color': hair_color_list,
        'gender': gender_list,
        'earring': earring_list,
        'smile': smile_list,
        'frontal_face': frontal_face_list
    }
    out = pd.DataFrame(out_dict)
    out.to_csv('./result/result/{}.csv'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr) + '_v1'), index=False)


def test_v2(model_name):
    test_iter = dataloader(test_json_path, data_dir, batch_size, False)

    model = get_model(model_name, 'v2')
    model.load_state_dict(torch.load('./result/model/{}.pth'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr)+'_v2')
    ))
    model.to(device)
    model.eval()

    hair_list = []
    gender_list = []
    earring_list = []
    smile_list = []
    frontal_face_list = []

    with tqdm(test_iter, unit='batch') as tepoch:
        total_acc = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = 0.
        for data, label in tepoch:
            tepoch.set_description(f"test")
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)

            acc1 += (outputs[0].argmax(1) == label[:, 0]).sum().item() / len(test_iter) / batch_size
            for i in outputs[0].argmax(1):
                hair_list.append(i.item())

            acc2 += (outputs[1].argmax(1) == label[:, 2]).sum().item() / len(test_iter) / batch_size
            for i in outputs[1].argmax(1):
                gender_list.append(i.item())

            acc3 += (outputs[2].argmax(1) == label[:, 3]).sum().item() / len(test_iter) / batch_size
            for i in outputs[2].argmax(1):
                earring_list.append(i.item())

            acc4 += (outputs[3].argmax(1) == label[:, 4]).sum().item() / len(test_iter) / batch_size
            for i in outputs[3].argmax(1):
                smile_list.append(i.item())

            acc5 += (outputs[4].argmax(1) == label[:, 5]).sum().item() / len(test_iter) / batch_size
            for i in outputs[4].argmax(1):
                frontal_face_list.append(i.item())

            total_acc = (acc1 + acc2 + acc3 + acc4 + acc5) / 5
            tepoch.set_postfix(total_acc=total_acc)

    out_dict = {
        'hair': hair_list,
        'gender': gender_list,
        'earring': earring_list,
        'smile': smile_list,
        'frontal_face': frontal_face_list
    }
    out = pd.DataFrame(out_dict)
    out.to_csv('./result/result/{}.csv'.format(
        model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(lr) + '_v2'
    ), index=False)

def test(model_name, model_version):
    if model_version == 'v1':
        test_v1(model_name)
    elif model_version == 'v2':
        test_v2(model_name)

