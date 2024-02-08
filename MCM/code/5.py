from pyecharts import options as opts
from pyecharts.charts import Pie
# 虚假数据
data_pair = [['Klein4000', 13903.4], ['MEPUS-AUV3000L', 20855.1], ['Falcon-DR', 10427.6]]

chart = Pie()
chart.add(
       '', 
       data_pair,
       # 仅通过扇形的ban
       rosetype='area'
       )
       
chart.render_notebook()