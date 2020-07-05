from project import Project
from tools.parser import Parser


dfs = Parser.parse()
project = Project(dfs)
