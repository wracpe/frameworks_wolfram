import matplotlib.pyplot as plt

import settings as sett

from project import Project
from well import Well


class Plotter(object):

    plt.rcParams.update({'figure.figsize': (20, 10), 'figure.dpi': 100})
    _save_directory = sett.work_path / sett.save_folder

    @classmethod
    def create_well_plot(cls, well: Well):
        fig, axs = plt.subplots(nrows=2, ncols=1)
        ax = axs[0]
        well.df.plot(grid=True, ax=ax)
        well.y_adap.plot(label=f'{sett.predicate}_адаптация', ax=ax)
        well.y_fore.plot(label=f'{sett.predicate}_прогноз', ax=ax)
        ax = axs[1]
        well.y_dev.plot(label='Отн. отклонение дебита жидкости от факта, %', ax=ax)
        plt.grid()
        plt.legend()
        plt.savefig(fname=cls._save_directory / f'{well.name}.png')

    @classmethod
    def create_project_plot(cls, project: Project):
        plt.figure()
        x = project.y_dev.index
        y = project.y_dev.to_list()
        plt.plot(x, y, label='Сред. отн. отклонение дебита жидкости от факта, %')
        plt.grid()
        plt.legend()
        plt.savefig(fname=cls._save_directory / 'performance.png')
