#make a bokeh scatterplot from popvae output
import bokeh, pandas as pd, numpy as np
from bokeh import models, plotting, io
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--latent_coords",help="path to latent_coords.txt file")
parser.add_argument("--metadata",help="path to metadata file with column named 'sampleID'")
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
meta=pd.read_csv(metadata,sep="\t")
ld=ld.merge(meta,on="sampleID")

if not colorby==None:
    if len(pd.factorize(ld[colorby])[1])>16:
        print("color column must have <16 unique values")
        exit()

#plot
p = bokeh.plotting.figure(plot_width=900, plot_height=600,
                          x_axis_label='LD1',
                          y_axis_label='LD2')

if not colorby==None:
    pal=["#A6CEE3","#1F78B4","#B2DF8A","#33A02C","#FB9A99","#E31A1C",
         "#FDBF6F","#FF7F00","#CAB2D6","#6A3D9A","#FFFF99","#B15928",
         "#d442f5","#403f40","#291dad"]
    keys=np.array(pd.factorize(ld[colorby])[1])
    cmap={keys[x]:pal[x] for x in range(len(pd.factorize(ld[colorby])[1]))}
    for key, group in ld.groupby(colorby):
        p.circle(x='mean1',y='mean2',size=7,alpha=0.75,
                 source=bokeh.models.ColumnDataSource(group),
                 color=cmap[key],legend=key)
else:
    p.circle(x="mean1",y="mean2",size=7,alpha=0.75,source=ld,color="black")

#hover panels
tooltips=[(x,'@'+x) for x in ld.columns]
hover = bokeh.models.HoverTool(tooltips=tooltips)
p.add_tools(hover)

if not outfile==None:
    bokeh.io.output_file(filename=outfile,mode="inline") #outputs an html
bokeh.io.show(p)
bokeh.io.reset_output()
