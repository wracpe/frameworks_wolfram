import settings as sett

from project import Project
from tools.data_handler import read_prepare
from tools.plotter import Plotter


well_data = read_prepare()
project = Project(well_data)


for well in project.wells:
    Plotter.create_well_plot(well)

Plotter.create_project_plot(project)
