def dataset_parser(dataset):
    c_list = list()
    a_list = list()
    m_list = list()

    for key in dataset.keys():
        params = key.split('__')
        for p in params:
            p_name, p_value = p.split('-')

            if p_name == 'class':
                c_list.append(p_value)
            elif p_name == 'action':
                a_list.append(p_value)
            elif p_name == 'morphology':
                m_list.append(p_value)

    c_list = sorted(list(set(c_list)))
    a_list = sorted(list(set(a_list)))
    m_list = sorted(list(set(m_list)))

    return {
        'c_list': c_list,
        'a_list': a_list,
        'm_list': m_list,
    }

# 今回は、分散が0のデータを取り除きます。
# また取り除いたデータと同じクラスのデータも取り除きます。
def remove_invalid_data(train_dataset):

    # sensor_valuesが1なら分散は0
    invalid_c_list = []
    for k, v in train_dataset.items():
        if len(v['sensor_values']) == 1:
            invalid_c = k.split('__')[0]
            invalid_c_list.append(invalid_c)

    invalid_c_list = list(set(invalid_c_list))

    # 取り除くデータと同じクラスのデータをリストアップ
    invalid_key_list = []
    for k in train_dataset.keys():
        k = str(k)
        for invalid_c in invalid_c_list:
            if k.startswith(invalid_c):
                invalid_key_list.append(k)

    invalid_key_list = list(set(invalid_key_list))

    # Training Dataから、上記のデータを削除
    for invalid_key in invalid_key_list:
        del train_dataset[invalid_key]

    return train_dataset
