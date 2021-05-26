from config_field import ConfigField
from _wrapper_field import _WrapperField


# ВНИМАНИЕ! УКАЗАНЫ ОПТИМАЛЬНЫЕ ЗНАЧЕНИЯ ПО ПАРАМЕТРАМ WINDOW_SIZES И QUANTILES.
'''КРАЙНЕЕ---------------------------------------------------------------------------
'''
config_kraynee_liq = ConfigField(
    name='Крайнее',
    predicate='ql_m3_fact',
    estimator_name_field='xgb',
    estimator_name_well='svr',
    is_deep_grid_search=False,
    window_sizes=[3, 7, 15, 30],
    quantiles=[0.1, 0.3],
    # well_names_ois=[
    #     '2560006108',
    # ],
)
config_kraynee_oil = ConfigField(
    name='Крайнее',
    predicate='qo_m3_fact',
    estimator_name_field='xgb',
    estimator_name_well='svr',
    is_deep_grid_search=False,
    window_sizes=[3, 7, 15, 30],
    quantiles=[0.1, 0.3],
    # well_names_ois=[
    #     '2560006108',
    # ],
)
'''ВАЛЫНТОЙСКОЕ---------------------------------------------------------------------------
Скважины:
9350049700
9350049800
не считаются, т.к. маленькая история для обучения.
'''
config_valyntoyskoe_liq = ConfigField(
    name='Валынтойское',
    predicate='ql_m3_fact',
    estimator_name_field='xgb',
    estimator_name_well='svr',
    is_deep_grid_search=False,
    window_sizes=[3, 7, 15, 30],
    quantiles=[0.1, 0.3],
    # well_names_ois=[
    #     '9350040000',
    # ],
)
config_valyntoyskoe_oil = ConfigField(
    name='Валынтойское',
    predicate='qo_m3_fact',
    estimator_name_field='xgb',
    estimator_name_well='svr',
    is_deep_grid_search=False,
    window_sizes=[3, 7, 15, 30],
    quantiles=[0.1, 0.3],
    # well_names_ois=[
    #     '9350040000',
    # ],
)
'''ВЫНГАЯХИНСКОЕ---------------------------------------------------------------------------
'''
config_vyngayakhinskoe_liq = ConfigField(
    name='Вынгаяхинское',
    predicate='ql_m3_fact',
    estimator_name_field='ela',
    estimator_name_well='svr',
    is_deep_grid_search=False,
    window_sizes=[3, 7, 15, 30],
    quantiles=[0.1, 0.3],
    # well_names_ois=[
    #     '9350049000',
    # ],
)
config_vyngayakhinskoe_oil = ConfigField(
    name='Вынгаяхинское',
    predicate='qo_m3_fact',
    estimator_name_field='ela',
    estimator_name_well='svr',
    is_deep_grid_search=False,
    window_sizes=[3, 7, 15, 30],
    quantiles=[0.1, 0.3],
    # well_names_ois=[
    #     '2860077600',
    # ],
)
configs = [
    # config_kraynee_liq,
    # config_kraynee_oil,
    config_valyntoyskoe_liq,
    config_valyntoyskoe_oil,
    # config_vyngayakhinskoe_liq,
    # config_vyngayakhinskoe_oil,
]
for config in configs:
    print(f'Calculation is started for "{config.name}" by "{config.predicate}".')
    _WrapperField(config)
