import warnings

from project import Project
from tools.parser import Parser
from tools.plotter import Plotter

warnings.filterwarnings('ignore')

dfs = Parser.parse()
project = Project(dfs)
for well in project.wells:
    Plotter.create_well_plot(well)
Plotter.create_project_plot(project)
