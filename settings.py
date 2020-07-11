import pathlib


work_path = pathlib.Path.cwd()
file_name = 'day.csv'
save_folder = 'plots'

forecast_days_number = 90
predicate = 'Дебит жидкости среднесуточный'
predictor = 'Давление забойное от Pпр'

usable_columns = ['Дата',
                  'Время работы (ТМ)',
                  predictor,
                  predicate]  # Don't remove 'Дата' from this list

window_sizes = [4, 5, 10, 20, 30, 40]
quantiles = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
