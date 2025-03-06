import numpy as np
import matplotlib.pyplot as plt
import fibrosis_model as fm
import senescence_model as sm

class Plotter:
    def __init__(self,simulation):
        self.simulation = simulation
    
    def plot_scatter(self,points,pointlabels=None,axlabels = None,colours=None,log=[False,False],lims=[None,None], show=True):
        """
        Input:
        points: List or array of (x,y) to be plotted
        pointlabels: List or array of labels for each point (optional)
        axlabels: Axes labels, pass as an list or array of both x and y (optional)
        log: List of Booleans, for x-axis and y-axis respectively, whether to use log-scales or not (optional, default is False,False)
        lims: List of limits for x and y
        show: Boolean, whether to show the plot or not (optional, default is True)
        """
        if pointlabels is not None and len(pointlabels) != len(points):
            raise ValueError("If using labels for points, make sure each point has a label")
        for i in range(len(points)):
            if pointlabels is not None and colours is None:
                plt.scatter(points[i][1],points[i][0],label=pointlabels[i])
            elif pointlabels is None and colours is not None:
                plt.scatter(points[i][1],points[i][0],color=colours[i])
            elif pointlabels is not None and colours is not None:
                plt.scatter(points[i][1],points[i][0],label=pointlabels[i],color=colours[i])
            else:
                plt.scatter(points[i][1],points[i][0])
        if axlabels != None:
            plt.xlabel(axlabels[0])
            plt.ylabel(axlabels[1])
        if log[0] == True:
            plt.xscale('log')
        if log[1] == True:
            plt.yscale('log')
        if lims[0] != None:
            plt.xlim((lims[0][0],lims[0][1]))
        if lims[1] != None:
            plt.ylim((lims[1][0],lims[1][1]))

        if show == True:
            plt.show()

    def plot_series(self,points,pointlabels=None,axlabels=None,log=[False,False],lims=[None,None],colours = None,show=True):
        """        Input:
        points: List or array of (x,y) to be plotted at each point
        pointlabels: List or array of labels for each series (optional)
        axlabels: Axes labels, pass as an list or array of both x and y (optional)
        log: List of Booleans, for x-axis and y-axis respectively, whether to use log-scales or not (optional, default is False,False)
        show: Boolean, whether to show the plot or not (optional, default is True)
        """
        if pointlabels is not None and len(pointlabels) != len(points):
            raise ValueError("If using labels for points, make sure each point has a label")
        for i in range(len(points)):
            if pointlabels is not None and colours is None:
                plt.plot(points[i][0],points[i][1],label=pointlabels[i])
            elif pointlabels is None and colours is not None:
                plt.plot(points[i][0],points[i][1],color=colours[i])
            elif pointlabels is not None and colours is not None:
                plt.plot(points[i][0],points[i][1],label=pointlabels[i],color=colours[i])
            else:
                plt.plot(points[i][0],points[i][1])
        if axlabels != None:
            plt.xlabel(axlabels[0])
            plt.ylabel(axlabels[1])
        if log[0] == True:
            plt.xscale('log')
        if log[1] == True:
            plt.yscale('log')
        if lims[0] != None:
            plt.xlim((lims[0][0],lims[0][1]))
        if lims[1] != None:
            plt.ylim((lims[1][0],lims[1][1]))

        if show == True:
            plt.show()
