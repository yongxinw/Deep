import matplotlib

from PIL import Image
from pylab import *
import matplotlib.pyplot as plt


def fig2img(fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    im = Image.frombytes("RGBA", (w, h), buf.tostring())
    return im


def fig2imgarr(fig):
    im = fig2img(fig)
    imarr = np.asarray(im)
    imarr = np.delete(imarr, 3, 2)
    return imarr


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def sphere_line_plot(lines, size, alpha=0.1, f=1.0, alternative=False):
    matplotlib.use('Agg')
    plt.ioff()	

    a = linspace(-pi / 2, pi / 2, num=10000)

    fig = plt.figure(figsize=(size/100.0, size/100.0), dpi=100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-pi / 2, pi / 2, -pi / 2, pi / 2])

    fig.add_axes(ax)

    ax.set_facecolor((0, 0, 0))

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for i in range(lines.shape[0]):
        lines[i, 0] *= f
        lines[i, 1] *= f

        if alternative:
            b = -np.arctan(-lines[i, 2] / (np.cos(a) * lines[i,0] + np.sin(a) * lines[i,1]))
        else:
            b = -np.arctan((-lines[i, 0] * np.sin(a) - lines[i, 2] * np.cos(a)) / lines[i, 1])

        b *= -1

        ax.plot(a, b, '-', c=[1, 1, 1, alpha])

    img = fig2imgarr(fig)
    imgray = np.mean(img, axis=2).astype(uint8)

    plt.close(fig)
    matplotlib.use('TkAgg')
    return imgray


def sphere_line_plot_image(lines, size, alpha=0.5, f=1.0, alternative=False):
    imgray = sphere_line_plot(lines, size, alpha, f, alternative)
    ims = Image.fromarray(imgray, 'L')
    return ims


def plot_sphere_lines_to_tile(lines, im_size, filename, alpha=0.5, f=1.0, alternative=False):
    imsr = sphere_line_plot_image(lines, im_size, alpha, f, alternative)
    imsr.save(filename)


def makeImage(pointpairs, size):

    fig = plt.figure(figsize=(size/50.0, size/50.0), dpi=50.0)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-1, 1, -1, 1])
    ax.set_axis_bgcolor((0, 0, 0))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    fig.add_axes(ax)

    for i in range(pointpairs.shape[0]):
        pp = np.squeeze(pointpairs[i, :])

        ax.plot([pp[0], pp[2]], [pp[1], pp[3]], 'w-')

    img = fig2imgarr(fig)
    imgray = np.mean(img, axis=2).astype(np.uint8)
    return imgray


