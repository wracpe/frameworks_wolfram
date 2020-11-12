import matplotlib.pyplot as plt
import pandas as pd

import settings as sett

from project import Project
from well import Well


class Plotter(object):

    plt.rcParams.update({'figure.figsize': (20, 10), 'figure.dpi': 100})
    _save_directory = sett.save_path

    @classmethod
    def create_well_plot(cls, well: Well):
        fig = plt.Figure()

        ax_1 = fig.add_subplot(321)
        y_1_fact = pd.concat(objs=[well.data['y_train'], well.data['y_test']])
        y_1_model = pd.concat(objs=[well.y_adap, well.y_pred])
        y_1_fact.plot(ax=ax_1)
        y_1_model.plot(ax=ax_1)
        ax_1.grid()
        ax_1.legend()

        ax_2 = fig.add_subplot(323, sharex=ax_1)
        y_2 = pd.concat(objs=[well.data['x_train'][sett.predictor], well.data['x_test'][sett.predictor]])
        y_2.plot(ax=ax_2)
        ax_2.grid()
        ax_2.legend()

        ax_3 = fig.add_subplot(325, sharex=ax_1)
        y_3 = pd.concat(objs=[well.data['x_train']['Время работы (ТМ)'], well.data['x_test']['Время работы (ТМ)']])
        y_3.plot(ax=ax_3)
        ax_3.grid()
        ax_3.legend()

        ax_4 = fig.add_subplot(222)
        y_4_fact = well.data['y_test']
        y_4_model = well.y_pred
        y_4_fact.plot(ax=ax_4)
        y_4_model.plot(ax=ax_4)
        ax_4.grid()
        ax_4.legend()

        ax_5 = fig.add_subplot(326, sharex=ax_4)
        y_5 = well.y_dev
        y_5.plot(ax=ax_5)
        ax_5.grid()
        ax_5.legend()

        # well.data['x_train']['Давление забойное от Pпр'].plot(ax=ax)
        # well.data['x_test']['Давление забойное от Pпр'].plot(ax=ax)
        # well.data['x_train']['Время работы (ТМ)'].plot(ax=ax)
        # well.data['x_test']['Время работы (ТМ)'].plot(ax=ax)
        # ax.grid()
        # ax.legend()
        #
        # ax = axs[1]
        # well.y_dev.plot(label='Отн. отклонение дебита жидкости от факта, %', ax=ax)
        # ax.grid()
        # ax.legend()

        fig.savefig(fname=cls._save_directory / f'{well.name}.png')

    @classmethod
    def create_project_plot(cls, project: Project):
        plt.figure()
        x = project.y_dev.index
        y = project.y_dev.to_list()
        plt.plot(x, y, label='Сред. отн. отклонение дебита жидкости от факта, %')
        plt.grid()
        plt.legend()
        plt.savefig(fname=cls._save_directory / 'performance.png')
