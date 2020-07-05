from tools.parser import Parser
from tools.plotter import Plotter
from well import Well


dfs = Parser.parse()

wells = []
for well_name, df in dfs.items():
    well = Well(well_name, df)
    Plotter.create(well)
    wells.append(well)
