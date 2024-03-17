'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Definições de cores -> todas estão numa escala de mais escura para mais clara.
VERMELHO_FORTE = '#461220'
CINZA1, CINZA2, CINZA3 = '#231F20', '#414040', '#555655'
CINZA4, CINZA5, CINZA6 = '#646369', '#76787B', '#828282'
CINZA7, CINZA8, CINZA9 = '#929497', '#A6A6A5', '#BFBEBE'
AZUL1, AZUL2, AZUL3, AZUL4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
VERMELHO1, VERMELHO2, VERMELHO3, VERMELHO4, VERMELHO5 = '#DB0527', '#E23652', '#ED8293', '#F4B4BE', '#FBE6E9'
VERDE1, VERDE2 = '#0C8040', '#9ABB59'
LARANJA1 = '#F79747'
AMARELO1, AMARELO2, AMARELO3, AMARELO4, AMARELO5 = '#FFC700', '#FFCC19', '#FFEB51', '#FFE37F', '#FFEEB2'
BRANCO = '#FFFFFF'


def barh_plot(data, y, width, title, ytick_range, 
              bar_color=AZUL1, text_color=CINZA1, 
              ytick_color=CINZA1, text_space=0.5, 
              title_color=CINZA1, title_fontsize=12, 
              title_loc='left', text_size=10.4, 
              ticksize=10.4, figsize=(6.4, 4.8), 
              invert_yaxis=False, mean=False):
    '''
    Plot a horizontal bar chart.
    
    This function plots a horizontal bar chart based on the provided data.
    
    Args:
        data (DataFrame): The DataFrame containing the data to be plotted.
        y (str): The name of the column in the DataFrame representing the categories for the y-axis.
        width (str): The name of the column in the DataFrame representing the width of the bars.
        title (str): The title of the plot.
        ytick_range (array-like): The range of values for y-axis ticks.
        bar_color (str, optional): The color of the bars. Defaults to AZUL1.
        text_color (str, optional): The color of the text annotations. Defaults to CINZA1.
        ytick_color (str, optional): The color of the y-axis ticks. Defaults to CINZA1.
        text_space (float, optional): The space between the bars and the text annotations. Defaults to 0.5.
        title_color (str, optional): The color of the title. Defaults to CINZA1.
        title_fontsize (int, optional): The font size of the title. Defaults to 12.
        title_loc (str, optional): The location of the title. Defaults to 'left'.
        text_size (float, optional): The font size of the text annotations. Defaults to 10.4.
        ticksize (float, optional): The font size of the y-axis ticks. Defaults to 10.4.
        figsize (tuple, optional): The size of the figure. Defaults to (6.4, 4.8).
        invert_yaxis (bool, optional): Whether to invert the y-axis or not. Defaults to False.
        mean (bool, optional): Whether to display mean values or not. Defaults to False.
    
    Returns:
        None
    
    Raises:
        CustomException: If an error occurs while plotting the horizontal bar chart.
    '''
    try:
        # Define the plot.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the horizontal bar plot.
        bars = ax.barh(y=data[y], width=data[width], color=bar_color)
        
        # Annotate the mean or the percentage.
        if mean:
            for index, value in enumerate(data[width]):
                ax.text(value + text_space, index, f'{round(value)}', va='center', color=text_color, fontsize=text_size)
                
        else:
            for index, value in enumerate(data[width]):
                ax.text(value + text_space, index, f'{value:.1f}%', va='center', color=text_color, fontsize=text_size)
        
        # Customize the plot.
        ax.set_title(title, color=title_color, fontweight='bold', loc=title_loc, fontsize=title_fontsize)
        ax.set_yticks(ticks=ytick_range, labels=data[y].tolist(), color=ytick_color, fontsize=ticksize)
        ax.tick_params(axis='y', which='both', left=False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)
        ax.get_xaxis().set_visible(False)
        
        if invert_yaxis:
            ax.invert_yaxis()
            
    except Exception as e:
        raise CustomException(e, sys)
    

def bar_plot(data, x, height, title, xlabel, labelpad=15, 
             labelcolor=CINZA1, title_color=CINZA1, spine_color=CINZA9,
             title_fontsize=12, title_pad=30, title_loc='left', 
             label_size=12, text_fontsize=10.5, bar_color=AZUL1, 
             barwidth=0.6, textcolor=BRANCO, figsize=(6.4, 4.8), 
             invert_xaxis=False, tick_size=10.4, top_annotation=False,
             top_space=0.5, mean=False):
    '''
    Plot a vertical bar chart.
    
    This function plots a vertical bar chart based on the provided data.
    
    Args:
        data (DataFrame): The DataFrame containing the data to be plotted.
        x (str): The name of the column in the DataFrame representing the categories for the x-axis.
        height (str): The name of the column in the DataFrame representing the height of the bars.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        labelpad (int, optional): The padding for the x-axis label. Defaults to 15.
        labelcolor (str, optional): The color of the x-axis label. Defaults to CINZA1.
        title_color (str, optional): The color of the title. Defaults to CINZA1.
        spine_color (str, optional): The color of the plot spines. Defaults to CINZA9.
        title_fontsize (int, optional): The font size of the title. Defaults to 12.
        title_pad (int, optional): The padding for the title. Defaults to 30.
        title_loc (str, optional): The location of the title. Defaults to 'left'.
        label_size (int, optional): The font size of the x-axis labels. Defaults to 12.
        text_fontsize (float, optional): The font size of the annotations. Defaults to 10.5.
        bar_color (str, optional): The color of the bars. Defaults to AZUL1.
        barwidth (float, optional): The width of the bars. Defaults to 0.6.
        textcolor (str, optional): The color of the annotations. Defaults to BRANCO.
        figsize (tuple, optional): The size of the figure. Defaults to (6.4, 4.8).
        invert_xaxis (bool, optional): Whether to invert the x-axis or not. Defaults to False.
        tick_size (float, optional): The font size of the x-axis ticks. Defaults to 10.4.
        top_annotation (bool, optional): Whether to annotate the top of the bars. Defaults to False.
        top_space (float, optional): The space between the annotation and the top of the bars. Defaults to 0.5.
        mean (bool, optional): Whether to display mean values or not. Defaults to False.
    
    Returns:
        None
        
    Raises:
        CustomException: If an error occurs while plotting the horizontal bar chart.
    '''
    try:
        # Define the plot.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the barplot.
        bars = ax.bar(x=data[x], height=data[height], color=bar_color, width=barwidth)
        
        # Customize the plot.
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=title_fontsize, pad=title_pad, loc=title_loc)
        ax.set_xlabel(xlabel, labelpad=labelpad, color=labelcolor, fontsize=label_size)
        ax.set_xticks(ticks=data[x], labels=data[x].unique().tolist(), color=labelcolor, fontsize=tick_size)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(spine_color)    
        ax.grid(False)
        
        # Annotate at the top of the bars, the percentage or the mean.
        if top_annotation:
            if mean:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(''.format(round(height)), 
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, top_space),  
                                textcoords="offset points",
                                ha='center', va='bottom', color=labelcolor,  
                                fontsize=text_fontsize)
            else:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate('{:.1f}%'.format(height), 
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, top_space),  
                                textcoords="offset points",
                                ha='center', va='bottom', color=labelcolor,  
                                fontsize=text_fontsize)
        
        # Annotate inside the bars, the proportion or the mean.
        else:
            if mean:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(round(height), 
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, -10),  
                                textcoords="offset points",
                                ha='center', va='center', color=textcolor,
                                fontsize=text_fontsize)
            else:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate('{:.1f}%'.format(height), 
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, -10),  
                                textcoords="offset points",
                                ha='center', va='center', color=textcolor,
                                fontsize=text_fontsize)
                
        if invert_xaxis:
            ax.invert_xaxis()
            
    except Exception as e:
        raise CustomException(e, sys)
    

def histogram_plot(data, x, title, xlabel, ylabel, stat='percent',
                   kde=True, hist_color=AZUL1, title_color=CINZA1, 
                   title_size=12, label_color=CINZA1, title_loc='left', 
                   y_label_loc='top', xlabel_loc='left', title_pad=20, 
                   add_ticks=100, labelpad=20, spines_ticks_colors=CINZA4,
                   bins='auto', figsize=(6.4, 4.8), label_size=10, tick_size=10):
    '''
    Plot a histogram.

    This function plots a histogram based on the provided data.

    Args:
        data (DataFrame): The DataFrame containing the data to be plotted.
        x (str): The name of the column in the DataFrame representing the data to be plotted on the x-axis.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        stat (str, optional): The type of statistic to compute. Defaults to 'percent'.
        kde (bool, optional): Whether to plot a kernel density estimate. Defaults to True.
        hist_color (str, optional): The color of the histogram bars. Defaults to AZUL1.
        title_color (str, optional): The color of the title text. Defaults to CINZA1.
        title_size (int, optional): The font size of the title. Defaults to 12.
        label_color (str, optional): The color of the axis labels. Defaults to CINZA1.
        title_loc (str, optional): The location of the title. Defaults to 'left'.
        y_label_loc (str, optional): The location of the y-axis label. Defaults to 'top'.
        xlabel_loc (str, optional): The location of the x-axis label. Defaults to 'left'.
        title_pad (int, optional): The padding of the title. Defaults to 20.
        add_ticks (int, optional): The interval for adding ticks on the x-axis. Defaults to 100.
        labelpad (int, optional): The padding of the axis labels. Defaults to 20.
        spines_ticks_colors (str, optional): The color of the spines and ticks. Defaults to CINZA4.
        bins (int, str or array, optional): Specification of histogram bins. Defaults to 'auto'.
        figsize (tuple, optional): The size of the figure. Defaults to (6.4, 4.8).
        label_size (int, optional): The font size of the axis labels. Defaults to 10.
        tick_size (int, optional): The font size of the axis ticks. Defaults to 10.

    Returns:
        None
        
    Raises:
        CustomException: If an error occurs while plotting the horizontal bar chart.
    '''
    try:
        # Define the plot.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the histogram.
        sns.histplot(data=data, x=x, kde=kde, color=hist_color, bins=bins, stat=stat)
        
        # Customize the plot.
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=title_size, loc=title_loc, pad=title_pad)
        ax.set_xlabel(xlabel, color=label_color, labelpad=labelpad, loc=xlabel_loc, fontsize=label_size)
        ax.set_ylabel(ylabel, color=label_color, labelpad=labelpad, loc=y_label_loc, fontsize=label_size)
        bin_edges = np.histogram_bin_edges(data[x], bins=bins)
        x_ticks = np.arange(bin_edges.min(), bin_edges.max() + add_ticks, add_ticks)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([int(x) for x in x_ticks], color=spines_ticks_colors, fontsize=tick_size)
        ax.set_yticklabels(ax.get_yticks(), fontsize=tick_size, color=spines_ticks_colors) 
        
        ax.spines['left'].set_color(spines_ticks_colors)
        ax.spines['bottom'].set_color(spines_ticks_colors)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        
    except Exception as e:
        raise CustomException(e, sys)
    

def univariate_boxplot(data, title, xlabel, ylabel, y_step, labelpad=20,
                       boxplot_color=CINZA1, label_color=CINZA1,
                       tick_spines_color=CINZA4, title_color=CINZA1, 
                       title_loc='left', pad=20, orient='v', y=None,
                       figsize=(6.4, 4.8), title_size=12, xaxis=True, 
                       ylabel_loc='top', label_size=10, tick_size=10):
    '''
    Plot a univariate boxplot.

    This function plots a univariate boxplot based on the provided data.

    Args:
        data (DataFrame): The DataFrame containing the data to be plotted.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        y_step (int): The step size for y-axis ticks.
        labelpad (int, optional): The padding of the axis labels. Defaults to 20.
        boxplot_color (str, optional): The color of the boxplot. Defaults to CINZA1.
        label_color (str, optional): The color of the axis labels. Defaults to CINZA1.
        tick_spines_color (str, optional): The color of the spines and ticks. Defaults to CINZA4.
        title_color (str, optional): The color of the title text. Defaults to CINZA1.
        title_loc (str, optional): The location of the title. Defaults to 'left'.
        pad (int, optional): The padding of the title. Defaults to 20.
        orient (str, optional): The orientation of the boxplot ('v' for vertical, 'h' for horizontal). Defaults to 'v'.
        y (str, optional): The name of the column in the DataFrame representing the data for the y-axis. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (6.4, 4.8).
        title_size (int, optional): The font size of the title. Defaults to 12.
        xaxis (bool, optional): Whether to display the x-axis or not. Defaults to True.
        ylabel_loc (str, optional): The location of the y-axis label. Defaults to 'top'.
        label_size (int, optional): The font size of the axis labels. Defaults to 10.
        tick_size (int, optional): The font size of the axis ticks. Defaults to 10.

    Returns:
        None
        
    Raises:
        CustomException: If an error occurs while plotting the horizontal bar chart.
    '''
    try:
        # Configure the plot.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the univariate boxplot.
        sns.boxplot(data=data, y=y, color=boxplot_color, orient=orient)
        
        # Customize the plot.
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=title_size, loc=title_loc, pad=pad)
        ax.set_xlabel(xlabel, color=label_color, labelpad=labelpad, fontsize=label_size)
        ax.set_ylabel(ylabel, color=label_color, labelpad=labelpad, loc=ylabel_loc, fontsize=label_size)
        ax.tick_params(axis='x', colors=tick_spines_color)
        ax.tick_params(axis='y', colors=tick_spines_color)
        
        ax.spines['left'].set_color(tick_spines_color)
        ax.spines['bottom'].set_color(tick_spines_color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
        
        y_max = data[y].max() if y is not None else max([max(subdata) for subdata in data])
        y_max = math.ceil(y_max)  
        y_ticks = list(range(0, y_max + y_step, y_step))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(y) for y in y_ticks], color=tick_spines_color, fontsize=tick_size)
        
        if not xaxis:
            ax.xaxis.set_visible(False)
    except Exception as e:
        raise CustomException(e, sys)
    
    
def bivariate_boxplot(data, title, xlabel, xtick_range, xtick_labels, 
                      ylabel, ytick_labels, y_step, labelpad=20,
                      boxplot_color=CINZA1, boxplot_palette=None, 
                      label_color=CINZA1, label_size=10, tick_size=10,
                      tick_spines_color=CINZA4, title_color=CINZA1, 
                      title_loc='left', order=None, pad=20, orient='v',
                      figsize=(6.4, 4.8), title_size=12, ylabel_loc='top',
                      xlabel_loc='left', x=None, y=None):
    '''
    Plot a bivariate boxplot.

    This function plots a bivariate boxplot based on the provided data.

    Args:
        data (DataFrame): The DataFrame containing the data to be plotted.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        xtick_range (list): The range of values for the x-axis ticks.
        xtick_labels (list): The labels for the x-axis ticks.
        ylabel (str): The label for the y-axis.
        ytick_labels (list): The labels for the y-axis ticks.
        y_step (int): The step size for y-axis ticks.
        labelpad (int, optional): The padding of the axis labels. Defaults to 20.
        boxplot_color (str, optional): The color of the boxplot. Defaults to CINZA1.
        boxplot_palette (str, optional): The color palette for the boxplot. Defaults to None.
        label_color (str, optional): The color of the axis labels. Defaults to CINZA1.
        label_size (int, optional): The font size of the axis labels. Defaults to 10.
        tick_size (int, optional): The font size of the axis ticks. Defaults to 10.
        tick_spines_color (str, optional): The color of the spines and ticks. Defaults to CINZA4.
        title_color (str, optional): The color of the title text. Defaults to CINZA1.
        title_loc (str, optional): The location of the title. Defaults to 'left'.
        order (list, optional): The order of the categories. Defaults to None.
        pad (int, optional): The padding of the title. Defaults to 20.
        orient (str, optional): The orientation of the boxplot ('v' for vertical, 'h' for horizontal). Defaults to 'v'.
        figsize (tuple, optional): The size of the figure. Defaults to (6.4, 4.8).
        title_size (int, optional): The font size of the title. Defaults to 12.
        ylabel_loc (str, optional): The location of the y-axis label. Defaults to 'top'.
        xlabel_loc (str, optional): The location of the x-axis label. Defaults to 'left'.
        x (str, optional): The name of the column in the DataFrame representing the data for the x-axis. Defaults to None.
        y (str, optional): The name of the column in the DataFrame representing the data for the y-axis. Defaults to None.

    Returns:
        None
        
    Raises:
        CustomException: If an error occurs while plotting the horizontal bar chart.
    '''
    try:
        # Define the plot.
        fig, ax = plt.subplots(figsize=figsize)
        
        if y:
            # Plot the boxplot when y is passed.
            sns.boxplot(data=data, x=x, y=y, orient=orient, palette=boxplot_palette, color=boxplot_color, order=order)

        else:
            # Plot the boxplot when y is passed.
            sns.boxplot(data=data, palette=boxplot_palette, color=boxplot_color, orient=orient, order=order)
        
        # Customize the plot.
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=title_size, loc=title_loc, pad=pad)
        ax.set_xlabel(xlabel, color=label_color, labelpad=labelpad, loc=xlabel_loc, fontsize=label_size)
        ax.set_ylabel(ylabel, color=label_color, labelpad=labelpad, loc=ylabel_loc, fontsize=label_size)
        ax.set_yticks(ticks=ytick_labels)
        ax.set_yticklabels(ytick_labels, fontsize=tick_size)
        ax.tick_params(axis='x', colors=tick_spines_color)
        ax.tick_params(axis='y', colors=tick_spines_color)
        ax.set_xticks(xtick_range)
        ax.set_xticklabels(xtick_labels, fontsize=tick_size)
        
        ax.spines['left'].set_color(tick_spines_color)
        ax.spines['bottom'].set_color(tick_spines_color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(False)
    except Exception as e:
        raise CustomException(e, sys)


def horizontal_stacked_bar_plot(data, first_x, second_x, y, y_tick_range, 
                                first_bar_label, second_bar_label, title, 
                                ytick_labels, first_bar_color=VERMELHO_FORTE, 
                                second_bar_color=CINZA8, ytick_color=CINZA1, 
                                ytick_size=10.8, text_size=10.8, text_color=CINZA1,
                                title_color=CINZA1, text_va='center', text_ha='left',
                                title_size=12, title_pad=20, order=None, 
                                figsize=(6.4, 4.8), bar_width=0.6, handles_x=0.1, 
                                handles_y=0.1, first_handle_color=VERMELHO_FORTE, 
                                second_handle_color=CINZA8, legend_x=0.16, legend_y=1.2, 
                                legend_size=10, frameon=False, title_loc='left'):
    '''
    Plot a horizontal stacked bar plot.

    This function plots a horizontal stacked bar plot based on the provided data.

    Args:
        data (DataFrame): The DataFrame containing the data to be plotted.
        first_x (str): The column name for the first set of bars on the x-axis.
        second_x (str): The column name for the second set of bars on the x-axis.
        y (str): The column name for the y-axis.
        y_tick_range (list): The range of values for the y-axis ticks.
        first_bar_label (str): The label for the first set of bars.
        second_bar_label (str): The label for the second set of bars.
        title (str): The title of the plot.
        ytick_labels (list): The labels for the y-axis ticks.
        first_bar_color (str, optional): The color of the first set of bars. Defaults to VERMELHO_FORTE.
        second_bar_color (str, optional): The color of the second set of bars. Defaults to CINZA8.
        ytick_color (str, optional): The color of the y-axis ticks. Defaults to CINZA1.
        ytick_size (float, optional): The font size of the y-axis ticks. Defaults to 10.8.
        text_size (float, optional): The font size of the text annotations. Defaults to 10.8.
        text_color (str, optional): The color of the text annotations. Defaults to CINZA1.
        title_color (str, optional): The color of the title text. Defaults to CINZA1.
        text_va (str, optional): The vertical alignment of the text annotations. Defaults to 'center'.
        text_ha (str, optional): The horizontal alignment of the text annotations. Defaults to 'left'.
        title_size (int, optional): The font size of the title. Defaults to 12.
        title_pad (int, optional): The padding of the title. Defaults to 20.
        order (list, optional): The order of the bars. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (6.4, 4.8).
        bar_width (float, optional): The width of the bars. Defaults to 0.6.
        handles_x (float, optional): The width of the legend handles. Defaults to 0.1.
        handles_y (float, optional): The height of the legend handles. Defaults to 0.1.
        first_handle_color (str, optional): The color of the legend handle for the first set of bars. Defaults to VERMELHO_FORTE.
        second_handle_color (str, optional): The color of the legend handle for the second set of bars. Defaults to CINZA8.
        legend_x (float, optional): The x-coordinate of the legend position. Defaults to 0.16.
        legend_y (float, optional): The y-coordinate of the legend position. Defaults to 1.2.
        legend_size (int, optional): The font size of the legend. Defaults to 10.
        frameon (bool, optional): Whether to draw a frame around the legend. Defaults to False.
        title_loc (str, optional): The location of the title. Defaults to 'left'.

    Returns:
        None
    
    Raises:
        CustomException: If an error occurs while plotting.
    '''
    try:
        # Configure the plot.
        fig, ax = plt.subplots(figsize=figsize)

        # Plot the two bars.
        sns.barplot(x=second_x, y=y, data=data, color=second_bar_color, label=second_bar_label, left=data[first_x], width=bar_width, order=order)
        sns.barplot(x=first_x, y=y, data=data, color=first_bar_color, label=first_bar_label, width=bar_width, order=order)
        
        # Customize the plot.
        ax.set_title(title, color=title_color, fontweight='bold', fontsize=title_size, pad=title_pad, loc=title_loc)
        ax.set_ylabel('')
        ax.set_yticks(ticks=y_tick_range, labels=ytick_labels, color=ytick_color, fontsize=ytick_size)
        ax.tick_params(axis='y', which='both', left=False)
        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(False)
        
        # Annotate the percentages inside the bars
        for i, bar in enumerate(ax.patches):
            width = bar.get_width()
            height = bar.get_height()
            x = bar.get_x()
            y = bar.get_y()
            
            if i < len(data):
                continue
            
            percentage = width * 100  
            ax.text(x + width / 2, y + height / 2, f"{percentage:.1f}%", ha=text_ha, va=text_va, color=text_color, fontsize=text_size)

        for i, bar in enumerate(ax.patches[:len(data)]):
            width = bar.get_width()
            height = bar.get_height()
            x = bar.get_x()
            y = bar.get_y()
            
            percentage = data[second_x].iloc[i] * 100  
            ax.text(x + width / 2, y + height / 2, f"{percentage:.1f}%", ha=text_ha, va=text_va, color=text_color, fontsize=text_size)
            
        # Define handles and labels for the legend with adjusted sizes
        handles = [plt.Rectangle((0,0),handles_x, handles_y, fc=first_handle_color, edgecolor = 'none'),
                plt.Rectangle((0,0), handles_x, handles_y, fc=second_handle_color, edgecolor = 'none')]
        labels = [first_bar_label, second_bar_label]

        # Add legend at the top of the first subplot with adjusted sizes and position
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=frameon, ncol=2, fontsize=legend_size)
    except Exception as e:
        raise CustomException(e, sys)
            