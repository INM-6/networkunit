try:
    from viziphant.rasterplot import rasterplot as vizi_rasterplot
    viziphant = True
except:
    viziphant = False


def rasterplot(spiketrains, **kwargs):
    if viziphant:
        return vizi_rasterplot(spiketrains, **kwargs)
    else:
        print('Missing viziphant package')