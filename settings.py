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

window_sizes = [5, 15, 25, 35, 45]
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
