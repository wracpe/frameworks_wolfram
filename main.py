from config_field import ConfigField
from _wrapper_field import _WrapperField


config_vyngayakhinskoe = ConfigField(
    name='Вынгаяхинское',
    target='ql_m3_fact',
    estimator_name_field='xgb',
    estimator_name_well='ela',
    is_deep_grid_search=True,
    window_sizes=[7],
    quantiles=[0.9],
    well_names_ois=[
        '2860077600',
    ],
)
wrapper_field = _WrapperField(config_vyngayakhinskoe)
