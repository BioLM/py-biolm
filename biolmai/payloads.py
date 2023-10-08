def INST_DAT_TXT(batch, include_batch_size=False):
    d = {"instances": []}
    for idx, row in batch.iterrows():
        inst = {"data": {"text": row.text}}
        d['instances'].append(inst)
    if include_batch_size is True:
        d['batch_size'] = len(d['instances'])
    return d
