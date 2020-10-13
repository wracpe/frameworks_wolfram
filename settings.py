import pathlib


work_path = pathlib.Path.cwd()
file_name = 'day.csv'
save_folder = 'plots'

forecast_days_number = 90
predicate = 'Дебит жидкости среднесуточный'
# predicate = 'Дебит нефти расчетный'
predictor = 'Давление забойное от Pпр'

usable_columns = ['Дата',
                  'Время работы (ТМ)',
                  predictor,
                  predicate]  # Don't remove 'Дата' from this list

# window_sizes = [3, 4, 5, 8, 10, 15, 20, 40]
window_sizes = [5]
# quantiles = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
quantiles = [0.9]
