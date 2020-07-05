import matplotlib.pyplot as plt

import settings as sett

from project import Project
from well import Well


class Plotter(object):

    _plot_directory = sett.work_path / sett.save_folder
    _dpi = 100

    @classmethod
    def create_well_plot(cls, well: Well):
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

        plot_directory = cls._plot_directory / f'{well.well_name}.png'
        plt.savefig(fname=plot_directory, dpi=cls._dpi)

    @classmethod
    def create_project_plot(cls, project: Project):
        project.y_dev.plot()
        plot_directory = cls._plot_directory / 'performance.png'
        plt.savefig(fname=plot_directory, dpi=cls._dpi)
