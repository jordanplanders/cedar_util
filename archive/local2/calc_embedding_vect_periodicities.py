

# Prepare embedding vectors

# calculate the significant periodicities of the embedding vectors

# write data to csv


def compute_psd(subset, var_name, label_prefix):
    """Compute the power spectral density (PSD) using Welch method."""
    ac_val = pyleo.utils.tsmodel.ar1_fit(subset[var_name])
    series = pyleo.Series(time=subset['date'], value=subset[var_name], value_name=f'{label_prefix}',
                          label=f'{ac_val:.3f}', time_name='years')
    # _, detrending = pyleo.utils.tsutils.detrend(series.value, method='emd')
    detrending = series.value - series.value.mean()
    detrended_series = pyleo.Series(time=series.time, value=detrending, value_name=f'{label_prefix}-trend',
                                    label=var_name, time_name='years')
    psd_welch = series.spectral(method='mtm')

    return series, detrended_series, psd_welch


def get_sig_periods(psd):
    psd_signif = psd.signif_test(number=1000, method='ar1sim')
    freq = psd_signif.frequency

    indexes_of_superior = np.argwhere(psd_signif.amplitude > psd_signif.signif_qs.psd_list[0].amplitude)
    freq_superior = freq[indexes_of_superior]
    return psd_signif, (1 / freq_superior).flatten()




if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Example usage of arguments
    # print(f"Number from script2: {args.number}")
    if args.project is not None:
        proj_name = args.project
    else:
        print('project name is required', file=sys.stdout, flush=True)
        print('project name is required', file=sys.stderr, flush=True)
        sys.exit(0)

    proj_dir = Path(os.getcwd()) / proj_name
    # with open(proj_dir / 'proj_config.yaml', 'r') as file:
    # config = yaml.safe_load(file)
    config = load_config(proj_dir / 'proj_config.yaml')

    data_dir = proj_dir / config.raw_data.name

    if Path('/Users/jlanders').exists() == True:
        calc_location = proj_dir / config.local.calc_carc  #'calc_local_tmp'
    else:
        calc_location = proj_dir / config.carc.calc_carc

    # may add flag specific things

    if args.flags is not None:
        flags = args.flags
    else:
        flags = ['raw_ts']

    # fig_dir = calc_location / config.raw_data.figures_dir
    # fig_dir.mkdir(exist_ok=True, parents=True)

    for author in
    arg_tuple = (config, data_dir, fig_dir, flags)

    # plot_raw(arg_tuple)
    embedding_explanation(arg_tuple)