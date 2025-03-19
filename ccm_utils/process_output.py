import pandas as pd


def unpack_ccm_output(CrossMapList_num):
    translate_d = {'columns': 'forcing', 'target': 'responding'}
    df_subs = []
    dta = CrossMapList_num['predictStats']

    for lib_size in dta.keys():
        df_sub = pd.DataFrame(dta[lib_size])
        df_sub['LibSize'] = lib_size  # Add the LibSize as a column
        df_subs.append(df_sub)
    df_sub = pd.concat(df_subs)

    # if isinstance(CrossMapList_num['columns'], list):
    #     forcing = ' '.join(CrossMapList_num['columns'])
    # else:
    #     forcing = CrossMapList_num['columns']
    responding = ' '.join(CrossMapList_num['columns']).strip('')
    forcing = ' '.join(CrossMapList_num['target']).strip('')

    df_sub['forcing'] =  forcing
    df_sub['responding'] = responding

    df_sub['relation'] = f'{forcing} causes {responding}'

    return df_sub

def add_meta_data(ccm_out, _ccm_out_df, train_ind_i, train_ind_f, lag=0, add_cols=None):
    _ccm_out_df['ind_i'] = train_ind_i
    _ccm_out_df['ind_f'] = train_ind_f
    _ccm_out_df['E'] = ccm_out.E
    _ccm_out_df['tau'] = ccm_out.tau
    _ccm_out_df['Tp'] = ccm_out.Tp
    _ccm_out_df['lag'] = lag
    if add_cols is not None:
        if not isinstance(add_cols, list):
            add_cols = [add_cols]
        for key in add_cols:
            if key == 'target':
                value = ' '.join(ccm_out.__dict__['target']).strip('')
            elif key == 'columns':
                value = ' '.join(ccm_out.__dict__['columns']).strip('')
            else:
                value =ccm_out.__dict__[key]
            _ccm_out_df[key] = value

    return _ccm_out_df