from config_field import ConfigField
from _wrapper_field import _WrapperField


config_vyngayakhinskoe = ConfigField(
    name='Вынгаяхинское',
    target='ql_m3_fact',
    estimator_name_field='ela',
    estimator_name_well='xgb',
    is_deep_grid_search=True,
    window_sizes=[3, 6, 12],
    quantiles=[0.2, 0.8],
    # well_names_ois=[
    #     '2860003808',
    # ],
)
wrapper_field = _WrapperField(config_vyngayakhinskoe)
