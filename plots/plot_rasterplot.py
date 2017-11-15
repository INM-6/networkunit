try:
    from viziphant.plots.rasterplot import rasterplot as vizi_rasterplot
    viziphant = True
except:
    viziphant = False

def rasterplot(spiketrains, **kwargs):
    if viziphant:
        vizi_rasterplot(spiketrains, **kwargs)
    else:
        print 'Missing viziphant package'