import xlwings as xw
import warnings

import settings as sett

from project import Project
from tools.parser import Parser
from tools.plotter import Plotter

warnings.filterwarnings('ignore')

well_data = Parser.parse()
project = Project(well_data)

# wb = xw.Book()

for well in project.wells:
    Plotter.create_well_plot(well)

    # wb.sheets.add(name=well.name)
    # sht = wb.sheets[f'{well.name}']
    #
    # sht.range((1, 1)).value = well.data['y_train'].loc[well.y_adap.index]
    # sht.range((1, 3)).value = well.y_adap
    # sht.range((1, 6)).value = well.data['y_test'].loc[well.y_pred.index]
    # sht.range((1, 8)).value = well.y_pred

# wb.sheets['Лист1'].delete()
# wb.save(path=sett.work_path / sett.save_folder / 'liq_prod_values.xlsx')
# wb.close()

Plotter.create_project_plot(project)
