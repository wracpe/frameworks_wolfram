from config_field import ConfigField
from wrapper_field import WrapperField


config_vyngayakhinskoe = ConfigField(
    name='Вынгаяхинское',
    target='ql_m3_fact',
    estimator_name_field='ela',
    estimator_name_well='xgb',
    is_deep_grid_search=True,
    window_sizes=[3, 6, 12],
    quantiles=[0.2, 0.8],
)
wrapper_field = WrapperField(config_vyngayakhinskoe)
