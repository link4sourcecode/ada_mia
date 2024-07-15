import torch

SEED = 2023

print("the selected random seed is: %s\t" % SEED)
print("--------------------------------------------------------------------------------")


def prepare_dataset_for_adamia_supervised(dataset):

    length = len(dataset)
    each_length = length//5
    # torch.manual_seed(SEED)
    data1, data2 = torch.utils.data.random_split(dataset, [each_length, len(dataset)-(each_length)])
    # data1, data2, data3, data4, data5 = torch.utils.data.random_split(
    #     dataset, [each_length, each_length, each_length, each_length, len(dataset) - (each_length * 4)])
    # return data1, data2, data3, data4, data5
    return data1, data2


def prepare_dataset(dataset):

    length = len(dataset)
    each_length = length//5
    # print(each_length)

    torch.manual_seed(0)
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    return target_train, target_test, shadow_train, shadow_test


def cut_dataset(dataset, num):

    length = len(dataset)

    torch.manual_seed(SEED)
    chosen_dataset, _ = torch.utils.data.random_split(
        dataset, [num, length - num])
    return chosen_dataset


def cut_dataset_random(dataset, num):

    length = len(dataset)

    # torch.manual_seed(SEED)
    chosen_dataset, _ = torch.utils.data.random_split(
        dataset, [num, length - num])
    return chosen_dataset

