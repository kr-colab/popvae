#make a bokeh scatterplot from popvae output
import bokeh, pandas as pd, numpy as np
from bokeh.models import Scatter
from bokeh.plotting import figure
from bokeh.io import show,reset_output
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--latent_coords",help="path to latent_coords.txt file")
parser.add_argument("--metadata",default=None,help="path to metadata file with column named 'sampleID'")
parser.add_argument("--colorby",default=None,help="<optional> column name for a metadata field to color points. \
                                      Must be character with < 16 unique values.")
parser.add_argument("--outfile",default=None,help="<optional> output path.")
args=parser.parse_args()

latent_coords=args.latent_coords
metadata=args.metadata
colorby=args.colorby
outfile=args.outfile

#load data
ld=pd.read_csv(latent_coords,sep="\t")
if metadata is not None:
    meta=pd.read_csv(metadata,sep="\t")
    ld=ld.merge(meta,on="sampleID")

if not colorby==None:
    if len(pd.factorize(ld[colorby])[1])>16:
        print("color column must have <16 unique values")
        exit()

#plot
p = figure(width=900, height=600,x_axis_label='LD1',y_axis_label='LD2')
if colorby==None:
    data = bokeh.models.ColumnDataSource(ld)
    glyph=Scatter(x='mean1',y='mean2',size=7,fill_alpha=0.75)
    p.add_glyph(data,glyph)

else:
    pal=["#A6CEE3","#1F78B4","#B2DF8A","#33A02C","#FB9A99","#E31A1C",
         "#FDBF6F","#FF7F00","#CAB2D6","#6A3D9A","#FFFF99","#B15928",
         "#d442f5","#403f40","#291dad"]
    keys=np.array(pd.factorize(ld[colorby])[1])
    cmap={keys[x]:pal[x] for x in range(len(pd.factorize(ld[colorby])[1]))}
    for key, group in ld.groupby(colorby):
        data = bokeh.models.ColumnDataSource(group)
        p.scatter(x='mean1',y='mean2',size=7,color=cmap[key],legend_label=key,source=data)
        #p.legend.location = "right"
        p.legend.title = colorby
        #glyph=Scatter(x='mean1',y='mean2',size=7,fill_alpha=0.75,fill_color=cmap[key],legend_label=key)
        #p.add_glyph(data,glyph)
        # p.circle(x='mean1',y='mean2',size=7,alpha=0.75,
        #          source=bokeh.models.ColumnDataSource(group),
        #          color=cmap[key],legend=key)

#hover panels
tooltips=[(x,'@'+x) for x in ld.columns]
hover = bokeh.models.HoverTool(tooltips=tooltips)
p.add_tools(hover)

if not outfile==None:
    bokeh.io.output_file(filename=outfile,mode="inline") #outputs an html
show(p)
reset_output()
