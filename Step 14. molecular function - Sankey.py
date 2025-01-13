import pandas as pd
import numpy as np

from pyecharts.charts import Sankey, Page, Grid, Graph
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot


# parameter
file_path = 'E:/PRAD/Step 1 bioinformatics process/Step 14. function research of DEM/'
save_path = 'E:/PRAD/Step 1 bioinformatics process/Step 14. function research of DEM/'

### load data
plot = pd.read_csv(file_path+'function.csv', encoding='utf-8')
plot['link'] = plot['type']+'_'+plot['Pathway']

### tidy data
print(plot.columns)
node_list = list(plot['type'].append(plot['Pathway']))
# count
set = np.array(plot['type'].append(plot['Pathway']))
set = np.unique(set)
node = list(set)
node_num = {}
for item in set:
    node_num.update({item:node_list.count(item)})

set_link = list(plot['link'])
link_dict = dict()
for item in set_link:
    if item in link_dict:
        link_dict[item] += 1
    else:
        link_dict[item] = 1

# nodes
nodes = []
for l in range(len(node)):
    t_dict = {}
    t_dict['name'] = node[l]
    nodes.append(t_dict)

# links
link = []
for i in range(plot.shape[0]):
    e = {}
    e['source'] = plot.iloc[i,1]
    e['target'] = plot.iloc[i,3]
    e['value'] = link_dict.get(plot.iloc[i,4])
    link.append(e)

sankey = (
    Sankey(init_opts=opts.InitOpts(width='900px',height='500px')).add(
        series_name='Molecular function Sankey plot',
        nodes=nodes,
        links=link,
        linestyle_opt=opts.LineStyleOpts(opacity=0.2,
                                         curve=0.5,
                                         color='source'),
        label_opts=opts.LabelOpts(font_size=14,
                                  position='left'),
        # levels=[opts.SankeyLevelsOpts(depth=0,
        #                               itemstyle_opts=opts.ItemStyleOpts(color='source'),
        #                               linestyle_opts=opts.LineStyleOpts(color='source', opacity=0.2, curve=0.5)),
        #         opts.SankeyLevelsOpts(depth=1,
        #                               itemstyle_opts=opts.ItemStyleOpts(color='#ff595e'),
        #                               linestyle_opts=opts.LineStyleOpts(color='source', opacity=0.2, curve=0.5))
        #         ]
    )
)
sankey.render(save_path + 'Molecular function Sankey plot.html')

## save other format
graph = Sankey(init_opts=opts.InitOpts(width='600px',height='500px')).add(
        series_name='Molecular function Sankey plot',
        nodes=nodes,
        links=link,
        linestyle_opt=opts.LineStyleOpts(opacity=0.2,
                                         curve=0.5,
                                         color='source'),
        label_opts=opts.LabelOpts(font_size=18,
                                  position='left',
                                  font_family = 'Times New Roman'),
        # levels=[opts.SankeyLevelsOpts(depth=0,
        #                               itemstyle_opts=opts.ItemStyleOpts(color='source'),
        #                               linestyle_opts=opts.LineStyleOpts(color='source', opacity=0.2, curve=0.5)),
        #         opts.SankeyLevelsOpts(depth=1,
        #                               itemstyle_opts=opts.ItemStyleOpts(color='#ff595e'),
        #                               linestyle_opts=opts.LineStyleOpts(color='source', opacity=0.2, curve=0.5))
        #         ]
    )
make_snapshot(snapshot, graph.render(), 'Molecular function Sankey plot.pdf')
