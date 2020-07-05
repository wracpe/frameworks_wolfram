import matplotlib.pyplot as plt

import settings as sett

from well import Well


class Plotter(object):

    @classmethod
    def create(cls, well: Well):
        columns = sett.usable_columns.copy()
        columns.remove('Дата')
        df = well.df[columns]

        df.plot(figsize=(25, 15),
                title=f'Скважина {well.well_name}',
                fontsize=14)

        well.y_adap.plot(label=f'{sett.predicate}_адаптация')
        well.y_fore.plot(label=f'{sett.predicate}_прогноз')
        plt.grid()
        plt.legend()

        plot_directory = sett.work_path / sett.save_folder / f'{well.well_name}.png'
        plt.savefig(fname=plot_directory, dpi=100)
