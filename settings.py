import pathlib


work_path = pathlib.Path.cwd()
data_path = work_path / 'data'
save_path = work_path / 'plots'

field = 'otdelnoe'

features_num = 5
forecast_days_number = 90
predicate = 'Дебит жидкости среднесуточный'  # Дебит жидкости среднесуточный, Дебит нефти расчетный
predictor = 'Давление забойное от Pпр'

usable_columns = ['Дата', 'Время работы (ТМ)', predictor, predicate]

# window_sizes = [3, 4, 5, 8, 10, 15, 20, 40]
window_sizes = [5]
# quantiles = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
quantiles = [0.9]
