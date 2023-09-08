def INST_DAT_TXT(batch):
    d = {"instances": []}
    for idx, row in batch.iterrows():
        inst = {"data": {"text": row.text}}
        d['instances'].append(inst)
    return d
