import matplotlib, numpy as np, math
import matplotlib.pyplot as plt

def imscale(signal, bounds):
    signal = np.array(signal)
    #replace -inf with zero
    signal[signal == -np.inf] = 0
    signal = np.nan_to_num(signal)
    low, high = (bounds[0], bounds[1])
    if low >= high:
        raise Exception("Bounds of imscale should be [low, high] where low < high")
    signal = signal - np.min(signal)
    return (signal / (np.max(signal) / (high-low))) + low

def imgmask(signal, mask):
    signal[signal < mask[0]] = mask[0]
    signal[signal > mask[1]] = mask[1]
    return signal

def plot1D(title, signal,  **kwargs):
    metadata = kwargs.get('metadata', None)
    fig, ax = (kwargs.get('fig', None), kwargs.get('ax', None))
    xtick_delta = kwargs.get('xtick_delta', 0.5)
    ylabel = kwargs.get('ylabel', None)
    ylim = kwargs.get('ylim', None)
    if ax == None:
        fig, ax = plt.subplots(figsize=(20,4))

    subplot = ax.plot(signal)
    ax.set_title(title)
    if ylim != None:
        ax.set_ylim(ylim)
    if metadata is not None:
        axes = image.get_axes()
        ptnsplit = eval(metadata['ptnsplit'])
        n_xticks = int(math.floor(metadata['duration'] / xtick_delta))
        xticks = [(tick * xtick_delta) / metadata['ptnblockwidth'] for tick in range(n_xticks)]
        axes.set_ylabel(ylabel)
        axes.set_xlabel('Time (s)')
        axes.set_xticks(xticks)
        axes.set_xticklabels(["{0:.1f}".format(t) for t in np.linspace(0, metadata['duration'], n_xticks)])

def plot2D(title, matrix, **kwargs):
    metadata = kwargs.get('metadata', None)
    fig, ax = (kwargs.get('fig', None), kwargs.get('ax', None))
    n_yticks = 10
    n_xticks = kwargs.get('xticks', 10)
    ylabel = kwargs.get('ylabel', None)
    scale = kwargs.get('scale', [0, 60])
    mask = kwargs.get('mask', [20,80])
    starttime = kwargs.get('starttime', 0)

    if ax == None:
        fig, ax = plt.subplots(figsize=(20,4))
    if scale is None:
        signal = matrix
    else:
        signal = imscale(matrix, scale)

    if mask is not None:
        signal = imgmask(signal, mask)

    if starttime > 0 and metadata == None:
        raise Exception('Cannot determine start frame from starttime without metadata')

    if starttime > 0:
        startframe = starttime / metadata['ptnblockwidth']
    else:
        startframe = 0

    image = ax.imshow(signal[:,startframe:], aspect='auto', interpolation='none', origin='bottom')
    fig.colorbar(image)
    ax.set_title(title)
    if metadata is not None:
        axes = image.get_axes()
        ptnsplit = eval(metadata['ptnsplit'])
        fMap = metadata['fMap'][ptnsplit[0]:ptnsplit[-1]]
        
        ytickvalues = [round(f)-ptnsplit[0] for f in np.linspace(ptnsplit[0], ptnsplit[-1]-1, n_yticks)]
        newyticks = ytickvalues
        newylabels = ["{}".format(int(round(fMap[f]/10)*10)) for f in newyticks]

        xtick_delta = metadata['duration'] / float(n_xticks)
        xticks = [(tick * xtick_delta) / metadata['ptnblockwidth'] for tick in range(starttime, n_xticks)]
        axes.set_ylabel('Frequency (Hz)')
        axes.set_yticks(newyticks)
        axes.set_yticklabels(newylabels)
        axes.set_xlabel('Time (s)')
        axes.set_xticks(xticks)
        axes.set_xticklabels(["{0:.1f}".format(t) for t in np.linspace(0, metadata['duration']-starttime, n_xticks)])
    plt.show()
